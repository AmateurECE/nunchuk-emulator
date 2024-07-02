use log::{info, warn};
use std::num::ParseIntError;
use std::sync::{Arc, RwLock};
use std::thread::spawn;

use thiserror::Error as ThisError;
use vhost::{vhost_user, vhost_user::Listener};
use vhost_user_backend::VhostUserDaemon;
use vm_memory::{GuestMemoryAtomic, GuestMemoryMmap};

use i2c::{I2cDevice, I2cMap, I2cReq, MAX_I2C_VDEV};
use once_cell::sync::Lazy;
use std::io::Write;
use vhu_i2c::VhostUserI2cBackend;

use crate::i2c;
use crate::vhu_i2c;

#[derive(Default, Debug)]
pub struct NunchukRegisters {
    z: bool,
    c: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Button {
    Z,
    C,
}

pub enum ButtonState {
    Unpressed,
    Pressed,
}

impl From<ButtonState> for bool {
    fn from(value: ButtonState) -> Self {
        match value {
            ButtonState::Unpressed => false,
            ButtonState::Pressed => true,
        }
    }
}

pub struct ButtonEvent {
    pub button: Button,
    pub state: ButtonState,
}

pub struct Nunchuk {
    adapter: u32,
    registers: Arc<RwLock<NunchukRegisters>>,
}

static GLOBAL_REGISTERS: Lazy<Arc<RwLock<NunchukRegisters>>> =
    Lazy::new(|| Arc::new(RwLock::new(NunchukRegisters::default())));
const NUNCHUK_I2C_ADDRESS: u16 = 0x52;
const NUNCHUK_TRANSFER_LENGTH: u16 = 6;

pub fn update_buttons(event: ButtonEvent) {
    let mut registers = GLOBAL_REGISTERS.write().unwrap();
    match event.button {
        Button::Z => registers.z = event.state.into(),
        Button::C => registers.c = event.state.into(),
    }
}

impl Nunchuk {
    fn dispatch_request(&self, request: &mut I2cReq) -> Result<(), i2c::Error> {
        if request.len == 0 {
            return Err(i2c::Error::I2cTransferInvalid(0));
        }

        if NUNCHUK_I2C_ADDRESS != request.addr {
            return Err(i2c::Error::ClientAddressInvalid);
        }

        match request.flags {
            // Attempt to write more than two bytes should fail
            0 if request.len > 2 => Err(i2c::Error::I2cTransferInvalid(request.len.into())),
            // Otherwise, writes are ignored (successfully)
            0 => Ok(()),
            // Read (in the normal case)
            1 if request.len == NUNCHUK_TRANSFER_LENGTH => request
                .buf
                .as_mut_slice()
                .write_all(&self.serialize_registers())
                .map_err(|_| i2c::Error::StdIoErr),
            // SMBUS reads, which may happen as a result of i2cdetect
            1 if request.len == 1 => request
                .buf
                .as_mut_slice()
                .write_all(&[0xff])
                .map_err(|_| i2c::Error::StdIoErr),
            // Other read operations fail.
            1 => Err(i2c::Error::I2cTransferInvalid(
                NUNCHUK_TRANSFER_LENGTH.into(),
            )),
            _ => Err(i2c::Error::StdIoErr),
        }
    }

    fn serialize_registers(&self) -> [u8; 6] {
        let registers = self.registers.read().unwrap();
        let mut last_byte: u8 = 0;
        if !registers.z {
            last_byte |= 1;
        }
        if !registers.c {
            last_byte |= 1 << 1;
        }

        [0, 0, 0, 0, 0, last_byte]
    }
}

impl i2c::I2cDevice for Nunchuk {
    // Open the device specified by the adapter name.
    fn open(adapter_name: &str) -> Result<Self, i2c::Error> {
        let parts: Vec<&str> = adapter_name.split('-').collect();
        let adapter = parts[1].parse::<u32>().unwrap();
        Ok(Nunchuk {
            adapter,
            registers: Arc::clone(&GLOBAL_REGISTERS),
        })
    }

    // Corresponds to the I2C_FUNCS ioctl call.
    fn funcs(&mut self) -> Result<u64, i2c::Error> {
        Ok(i2c::I2C_FUNC_I2C)
    }

    // Corresponds to the I2C_RDWR ioctl call.
    fn rdwr(&self, reqs: &mut [i2c::I2cReq]) -> Result<(), i2c::Error> {
        for request in reqs {
            self.dispatch_request(request)?;
        }

        Ok(())
    }

    // Corresponds to the I2C_SMBUS ioctl call.
    fn smbus(&self, _: &mut i2c::SmbusMsg) -> Result<(), i2c::Error> {
        Err(i2c::Error::AdapterFunctionInvalid(i2c::I2C_FUNC_SMBUS_ALL))
    }

    // Corresponds to the I2C_SLAVE ioctl call.
    fn slave(&self, _: u64) -> Result<(), i2c::Error> {
        Ok(())
    }

    // Returns the adapter number corresponding to this device.
    fn adapter_no(&self) -> u32 {
        self.adapter
    }
}

#[allow(dead_code)]
#[derive(Debug, PartialEq, ThisError)]
/// Errors related to low level i2c helpers
pub(crate) enum EmulatorError {
    #[error("Invalid socket count: {0}")]
    SocketCountInvalid(usize),
    #[error("Duplicate adapter detected: {0}")]
    AdapterDuplicate(String),
    #[error("Invalid client address: {0}")]
    ClientAddressInvalid(u16),
    #[error("Duplicate client address detected: {0}")]
    ClientAddressDuplicate(u16),
    #[error("Low level I2c failure: {0:?}")]
    I2cFailure(i2c::Error),
    #[error("Failed while parsing to integer: {0:?}")]
    ParseFailure(ParseIntError),
    #[error("Failed to join threads")]
    FailedJoiningThreads,
}

#[derive(Debug)]
pub struct I2cArgs {
    /// Location of vhost-user Unix domain socket. This is suffixed by 0,1,2..socket_count-1.
    pub socket_path: String,

    /// Number of guests (sockets) to connect to.
    pub socket_count: usize,

    /// List of I2C bus and clients in format
    /// <bus-name>:<client_addr>[:<client_addr>][,<bus-name>:<client_addr>[:<client_addr>]].
    pub device_list: String,
}

#[derive(Debug, PartialEq)]
pub(crate) struct DeviceConfig {
    pub(crate) adapter_name: String,
    pub(crate) addr: Vec<u16>,
}

impl DeviceConfig {
    fn new(name: &str) -> Result<Self, EmulatorError> {
        Ok(DeviceConfig {
            adapter_name: name.trim().to_string(),
            addr: Vec::new(),
        })
    }

    fn push(&mut self, addr: u16) -> Result<(), EmulatorError> {
        if addr as usize > MAX_I2C_VDEV {
            return Err(EmulatorError::ClientAddressInvalid(addr));
        }

        if self.addr.contains(&addr) {
            return Err(EmulatorError::ClientAddressDuplicate(addr));
        }

        self.addr.push(addr);
        Ok(())
    }
}

#[derive(Debug, PartialEq)]
pub(crate) struct AdapterConfig {
    pub(crate) inner: Vec<DeviceConfig>,
}

impl AdapterConfig {
    fn new() -> Self {
        Self { inner: Vec::new() }
    }

    fn contains_adapter(&self, adapter_name: &str) -> bool {
        self.inner
            .iter()
            .any(|elem| elem.adapter_name == adapter_name)
    }

    fn contains_addr(&self, addr: u16) -> bool {
        self.inner.iter().any(|elem| elem.addr.contains(&addr))
    }

    fn push(&mut self, device: DeviceConfig) -> Result<(), EmulatorError> {
        if self.contains_adapter(&device.adapter_name) {
            return Err(EmulatorError::AdapterDuplicate(device.adapter_name));
        }

        for addr in device.addr.iter() {
            if self.contains_addr(*addr) {
                return Err(EmulatorError::ClientAddressDuplicate(*addr));
            }
        }

        self.inner.push(device);
        Ok(())
    }
}

impl TryFrom<&str> for AdapterConfig {
    type Error = EmulatorError;

    fn try_from(list: &str) -> Result<Self, EmulatorError> {
        let busses: Vec<&str> = list.split(',').collect();
        let mut devices = AdapterConfig::new();

        for businfo in busses.iter() {
            let list: Vec<&str> = businfo.split(':').collect();
            let mut adapter = DeviceConfig::new(list[0])?;

            for device_str in list[1..].iter() {
                let addr = device_str
                    .parse::<u16>()
                    .map_err(EmulatorError::ParseFailure)?;
                adapter.push(addr)?;
            }

            devices.push(adapter)?;
        }
        Ok(devices)
    }
}

#[derive(PartialEq, Debug)]
struct I2cConfiguration {
    socket_path: String,
    socket_count: usize,
    devices: AdapterConfig,
}

impl TryFrom<I2cArgs> for I2cConfiguration {
    type Error = EmulatorError;

    fn try_from(args: I2cArgs) -> Result<Self, EmulatorError> {
        if args.socket_count == 0 {
            return Err(EmulatorError::SocketCountInvalid(0));
        }

        let devices = AdapterConfig::try_from(args.device_list.trim())?;
        Ok(I2cConfiguration {
            socket_path: args.socket_path.trim().to_string(),
            socket_count: args.socket_count,
            devices,
        })
    }
}

pub fn start_backend<D>(args: I2cArgs) -> Result<(), EmulatorError>
where
    D: 'static + I2cDevice + Send + Sync,
{
    let config = I2cConfiguration::try_from(args).unwrap();

    // The same i2c_map structure instance is shared between all the guests
    let i2c_map = Arc::new(I2cMap::<D>::new(&config.devices).map_err(EmulatorError::I2cFailure)?);

    let mut handles = Vec::new();

    for i in 0..config.socket_count {
        let socket = config.socket_path.to_owned() + &i.to_string();
        let i2c_map = i2c_map.clone();

        let handle = spawn(move || loop {
            // A separate thread is spawned for each socket and can connect to a separate guest.
            // These are run in an infinite loop to not require the daemon to be restarted once a
            // guest exits.
            //
            // There isn't much value in complicating code here to return an error from the
            // threads, and so the code uses unwrap() instead. The panic on a thread won't cause
            // trouble to other threads/guests or the main() function and should be safe for the
            // daemon.
            let backend = Arc::new(RwLock::new(
                VhostUserI2cBackend::new(i2c_map.clone()).unwrap(),
            ));
            let listener = Listener::new(socket.clone(), true).unwrap();

            let mut daemon = VhostUserDaemon::new(
                String::from("vhost-device-i2c-backend"),
                backend.clone(),
                GuestMemoryAtomic::new(GuestMemoryMmap::new()),
            )
            .unwrap();

            daemon.start(listener).unwrap();

            match daemon.wait() {
                Ok(()) => {
                    info!("Stopping cleanly.");
                }
                Err(vhost_user_backend::Error::HandleRequest(
                    vhost_user::Error::PartialMessage,
                )) => {
                    info!("vhost-user connection closed with partial message. If the VM is shutting down, this is expected behavior; otherwise, it might be a bug.");
                }
                Err(e) => {
                    warn!("Error running daemon: {:?}", e);
                }
            }

            // No matter the result, we need to shut down the worker thread.
            backend.read().unwrap().exit_event.write(1).unwrap();
        });

        handles.push(handle);
    }

    Ok(())
}
