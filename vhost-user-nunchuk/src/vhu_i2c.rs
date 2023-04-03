// vhost device i2c
//
// Copyright 2021 Linaro Ltd. All Rights Reserved.
//          Viresh Kumar <viresh.kumar@linaro.org>
//
// SPDX-License-Identifier: Apache-2.0 or BSD-3-Clause

use log::warn;
use std::mem::size_of;
use std::sync::Arc;
use std::{
    convert,
    io::{self, Result as IoResult},
};

use thiserror::Error as ThisError;
use vhost::vhost_user::message::{VhostUserProtocolFeatures, VhostUserVirtioFeatures};
use vhost_user_backend::{VhostUserBackendMut, VringRwLock, VringT};
use virtio_bindings::bindings::virtio_config::{VIRTIO_F_NOTIFY_ON_EMPTY, VIRTIO_F_VERSION_1};
use virtio_bindings::bindings::virtio_ring::{
    VIRTIO_RING_F_EVENT_IDX, VIRTIO_RING_F_INDIRECT_DESC,
};
use virtio_queue::{DescriptorChain, QueueOwnedT};
use vm_memory::{
    ByteValued, Bytes, GuestAddressSpace, GuestMemoryAtomic, GuestMemoryLoadGuard, GuestMemoryMmap,
    Le16, Le32,
};
use vmm_sys_util::epoll::EventSet;
use vmm_sys_util::eventfd::{EventFd, EFD_NONBLOCK};

use crate::i2c::*;

/// Virtio I2C Feature bits
const VIRTIO_I2C_F_ZERO_LENGTH_REQUEST: u16 = 0;

const QUEUE_SIZE: usize = 1024;
const NUM_QUEUES: usize = 1;

type Result<T> = std::result::Result<T, Error>;

#[derive(Copy, Clone, Debug, Eq, PartialEq, ThisError)]
/// Errors related to vhost-device-i2c daemon.
pub(crate) enum Error {
    #[error("Failed to handle event, didn't match EPOLLIN")]
    HandleEventNotEpollIn,
    #[error("Failed to handle unknown event")]
    HandleEventUnknown,
    #[error("Received unexpected write only descriptor at index {0}")]
    UnexpectedWriteOnlyDescriptor(usize),
    #[error("Received unexpected readable descriptor at index {0}")]
    UnexpectedReadableDescriptor(usize),
    #[error("Invalid descriptor count {0}")]
    UnexpectedDescriptorCount(usize),
    #[error("Invalid descriptor size, expected: {0}, found: {1}")]
    UnexpectedDescriptorSize(usize, u32),
    #[error("Descriptor not found")]
    DescriptorNotFound,
    #[error("Descriptor read failed")]
    DescriptorReadFailed,
    #[error("Descriptor write failed")]
    DescriptorWriteFailed,
    #[error("Failed to send notification")]
    NotificationFailed,
    #[error("Failed to create new EventFd")]
    EventFdFailed,
}

impl convert::From<Error> for io::Error {
    fn from(e: Error) -> Self {
        io::Error::new(io::ErrorKind::Other, e)
    }
}

/// I2C definitions from Virtio Spec

/// The final status written by the device
const VIRTIO_I2C_MSG_OK: u8 = 0;
const VIRTIO_I2C_MSG_ERR: u8 = 1;

#[derive(Copy, Clone, Default)]
#[repr(C)]
struct VirtioI2cOutHdr {
    addr: Le16,
    padding: Le16,
    flags: Le32,
}
// SAFETY: The layout of the structure is fixed and can be initialized by
// reading its content from byte array.
unsafe impl ByteValued for VirtioI2cOutHdr {}

/// VirtioI2cOutHdr Flags
const VIRTIO_I2C_FLAGS_M_RD: u32 = 1 << 1;

#[derive(Copy, Clone, Default)]
#[repr(C)]
struct VirtioI2cInHdr {
    status: u8,
}
// SAFETY: The layout of the structure is fixed and can be initialized by
// reading its content from byte array.
unsafe impl ByteValued for VirtioI2cInHdr {}

pub(crate) struct VhostUserI2cBackend<D: I2cDevice> {
    i2c_map: Arc<I2cMap<D>>,
    event_idx: bool,
    pub exit_event: EventFd,
    mem: Option<GuestMemoryLoadGuard<GuestMemoryMmap>>,
}

type I2cDescriptorChain = DescriptorChain<GuestMemoryLoadGuard<GuestMemoryMmap<()>>>;

impl<D: I2cDevice> VhostUserI2cBackend<D> {
    pub fn new(i2c_map: Arc<I2cMap<D>>) -> Result<Self> {
        Ok(VhostUserI2cBackend {
            i2c_map,
            event_idx: false,
            exit_event: EventFd::new(EFD_NONBLOCK).map_err(|_| Error::EventFdFailed)?,
            mem: None,
        })
    }

    /// Process the requests in the vring and dispatch replies
    fn process_requests(
        &self,
        requests: Vec<I2cDescriptorChain>,
        vring: &VringRwLock,
    ) -> Result<bool> {
        let mut reqs: Vec<I2cReq> = Vec::new();

        if requests.is_empty() {
            return Ok(true);
        }

        // Iterate over each I2C request and push it to "reqs" vector.
        for desc_chain in requests.clone() {
            let descriptors: Vec<_> = desc_chain.clone().collect();

            if (descriptors.len() != 2) && (descriptors.len() != 3) {
                return Err(Error::UnexpectedDescriptorCount(descriptors.len()));
            }

            let desc_out_hdr = descriptors[0];

            if desc_out_hdr.is_write_only() {
                return Err(Error::UnexpectedWriteOnlyDescriptor(0));
            }

            if desc_out_hdr.len() as usize != size_of::<VirtioI2cOutHdr>() {
                return Err(Error::UnexpectedDescriptorSize(
                    size_of::<VirtioI2cOutHdr>(),
                    desc_out_hdr.len(),
                ));
            }

            let out_hdr = desc_chain
                .memory()
                .read_obj::<VirtioI2cOutHdr>(desc_out_hdr.addr())
                .map_err(|_| Error::DescriptorReadFailed)?;

            let flags = match out_hdr.flags.to_native() & VIRTIO_I2C_FLAGS_M_RD {
                VIRTIO_I2C_FLAGS_M_RD => I2C_M_RD,
                _ => 0,
            };

            let desc_in_hdr = descriptors[descriptors.len() - 1];
            if !desc_in_hdr.is_write_only() {
                return Err(Error::UnexpectedReadableDescriptor(descriptors.len() - 1));
            }

            if desc_in_hdr.len() as usize != size_of::<u8>() {
                return Err(Error::UnexpectedDescriptorSize(
                    size_of::<u8>(),
                    desc_in_hdr.len(),
                ));
            }

            let (buf, len) = match descriptors.len() {
                // Buffer is available
                3 => {
                    let desc_buf = descriptors[1];
                    let len = desc_buf.len();

                    if len == 0 {
                        return Err(Error::UnexpectedDescriptorSize(1, len));
                    }
                    let mut buf = vec![0; len as usize];

                    if flags != I2C_M_RD {
                        if desc_buf.is_write_only() {
                            return Err(Error::UnexpectedWriteOnlyDescriptor(1));
                        }

                        desc_chain
                            .memory()
                            .read(&mut buf, desc_buf.addr())
                            .map_err(|_| Error::DescriptorReadFailed)?;
                    } else if !desc_buf.is_write_only() {
                        return Err(Error::UnexpectedReadableDescriptor(1));
                    }

                    (buf, len)
                }

                _ => (Vec::<u8>::new(), 0),
            };

            reqs.push(I2cReq {
                addr: out_hdr.addr.to_native() >> 1,
                flags,
                len: len as u16,
                buf,
            });
        }

        let in_hdr = {
            VirtioI2cInHdr {
                status: match self.i2c_map.transfer(&mut reqs) {
                    Ok(()) => VIRTIO_I2C_MSG_OK,
                    Err(_) => VIRTIO_I2C_MSG_ERR,
                },
            }
        };

        for (i, desc_chain) in requests.iter().enumerate() {
            let descriptors: Vec<_> = desc_chain.clone().collect();
            let desc_in_hdr = descriptors[descriptors.len() - 1];
            let mut len = size_of::<VirtioI2cInHdr>() as u32;

            if descriptors.len() == 3 {
                let desc_buf = descriptors[1];

                // Write the data read from the I2C device
                if reqs[i].flags == I2C_M_RD {
                    desc_chain
                        .memory()
                        .write(&reqs[i].buf, desc_buf.addr())
                        .map_err(|_| Error::DescriptorWriteFailed)?;
                }

                if in_hdr.status == VIRTIO_I2C_MSG_OK {
                    len += desc_buf.len();
                }
            }

            // Write the transfer status
            desc_chain
                .memory()
                .write_obj::<VirtioI2cInHdr>(in_hdr, desc_in_hdr.addr())
                .map_err(|_| Error::DescriptorWriteFailed)?;

            if vring.add_used(desc_chain.head_index(), len).is_err() {
                warn!("Couldn't return used descriptors to the ring");
            }
        }

        Ok(true)
    }

    /// Process the requests in the vring and dispatch replies
    fn process_queue(&self, vring: &VringRwLock) -> Result<bool> {
        let requests: Vec<_> = vring
            .get_mut()
            .get_queue_mut()
            .iter(self.mem.as_ref().unwrap().clone())
            .map_err(|_| Error::DescriptorNotFound)?
            .collect();

        if self.process_requests(requests, vring)? {
            // Send notification once all the requests are processed
            vring
                .signal_used_queue()
                .map_err(|_| Error::NotificationFailed)?;
        }

        Ok(true)
    }
}

/// VhostUserBackendMut trait methods
impl<D: 'static + I2cDevice + Sync + Send> VhostUserBackendMut<VringRwLock, ()>
    for VhostUserI2cBackend<D>
{
    fn num_queues(&self) -> usize {
        NUM_QUEUES
    }

    fn max_queue_size(&self) -> usize {
        QUEUE_SIZE
    }

    fn features(&self) -> u64 {
        // this matches the current libvhost defaults except VHOST_F_LOG_ALL
        1 << VIRTIO_F_VERSION_1
            | 1 << VIRTIO_F_NOTIFY_ON_EMPTY
            | 1 << VIRTIO_RING_F_INDIRECT_DESC
            | 1 << VIRTIO_RING_F_EVENT_IDX
            | 1 << VIRTIO_I2C_F_ZERO_LENGTH_REQUEST
            | VhostUserVirtioFeatures::PROTOCOL_FEATURES.bits()
    }

    fn protocol_features(&self) -> VhostUserProtocolFeatures {
        VhostUserProtocolFeatures::MQ
    }

    fn set_event_idx(&mut self, enabled: bool) {
        dbg!(self.event_idx = enabled);
    }

    fn update_memory(&mut self, mem: GuestMemoryAtomic<GuestMemoryMmap>) -> IoResult<()> {
        self.mem = Some(mem.memory());
        Ok(())
    }

    fn handle_event(
        &mut self,
        device_event: u16,
        evset: EventSet,
        vrings: &[VringRwLock],
        _thread_id: usize,
    ) -> IoResult<bool> {
        if evset != EventSet::IN {
            return Err(Error::HandleEventNotEpollIn.into());
        }

        match device_event {
            0 => {
                let vring = &vrings[0];

                if self.event_idx {
                    // vm-virtio's Queue implementation only checks avail_index
                    // once, so to properly support EVENT_IDX we need to keep
                    // calling process_queue() until it stops finding new
                    // requests on the queue.
                    loop {
                        vring.disable_notification().unwrap();
                        self.process_queue(vring)?;
                        if !vring.enable_notification().unwrap() {
                            break;
                        }
                    }
                } else {
                    // Without EVENT_IDX, a single call is enough.
                    self.process_queue(vring)?;
                }
            }

            _ => {
                warn!("unhandled device_event: {}", device_event);
                return Err(Error::HandleEventUnknown.into());
            }
        }
        Ok(false)
    }

    fn exit_event(&self, _thread_index: usize) -> Option<EventFd> {
        self.exit_event.try_clone().ok()
    }
}
