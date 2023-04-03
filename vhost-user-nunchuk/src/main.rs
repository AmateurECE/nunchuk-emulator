use std::{fs::File, os::unix::prelude::AsRawFd};
use wayland_client::{
    protocol::{
        wl_buffer, wl_compositor,
        wl_keyboard::{self, KeyState},
        wl_registry, wl_seat, wl_shm, wl_shm_pool, wl_surface,
    },
    Connection, Dispatch, QueueHandle, WEnum,
};
use wayland_protocols::xdg::shell::client::{xdg_surface, xdg_toplevel, xdg_wm_base};

use log::{info, warn};
use std::num::ParseIntError;
use std::sync::{Arc, RwLock};
use std::thread::spawn;

use clap::Parser;
use thiserror::Error as ThisError;
use vhost::{vhost_user, vhost_user::Listener};
use vhost_user_backend::VhostUserDaemon;
use vm_memory::{GuestMemoryAtomic, GuestMemoryMmap};

use i2c::{I2cDevice, I2cMap, I2cReq, MAX_I2C_VDEV};
use once_cell::sync::Lazy;
use vhu_i2c::VhostUserI2cBackend;

mod i2c;
mod vhu_i2c;

#[derive(Default, Debug)]
struct NunchukRegisters {
    z: bool,
    c: bool,
}

struct Nunchuk {
    adapter: u32,
    registers: Arc<RwLock<NunchukRegisters>>,
}

// This struct represents the state of our app. This simple app does not
// need any state, by this type still supports the `Dispatch` implementations.
#[derive(Default)]
struct State {
    base_surface: Option<wl_surface::WlSurface>,
    wm_base: Option<xdg_wm_base::XdgWmBase>,
    xdg_surface: Option<(xdg_surface::XdgSurface, xdg_toplevel::XdgToplevel)>,
    running: bool,
    buffer: Option<wl_buffer::WlBuffer>,
    configured: bool,
}

static GLOBAL_REGISTERS: Lazy<Arc<RwLock<NunchukRegisters>>> =
    Lazy::new(|| Arc::new(RwLock::new(NunchukRegisters::default())));
const NUNCHUK_I2C_ADDRESS: u16 = 0x52;
const NUNCHUK_TRANSFER_LENGTH: u16 = 6;

fn draw(tmp: &mut File, (buf_x, buf_y): (u32, u32)) {
    use std::{cmp::min, io::Write};
    let mut buf = std::io::BufWriter::new(tmp);
    for y in 0..buf_y {
        for x in 0..buf_x {
            let a = 0xFF;
            let r = min(((buf_x - x) * 0xFF) / buf_x, ((buf_y - y) * 0xFF) / buf_y);
            let g = min((x * 0xFF) / buf_x, ((buf_y - y) * 0xFF) / buf_y);
            let b = min(((buf_x - x) * 0xFF) / buf_x, (y * 0xFF) / buf_y);

            let color = (a << 24) + (r << 16) + (g << 8) + b;
            buf.write_all(&color.to_ne_bytes()).unwrap();
        }
    }
    buf.flush().unwrap();
}

impl Nunchuk {
    fn dispatch_request(&self, request: &mut I2cReq) -> Result<(), i2c::Error> {
        if request.len == 0 {
            return Err(i2c::Error::I2cTransferInvalid(0));
        }

        if NUNCHUK_I2C_ADDRESS != request.addr {
            return Err(i2c::Error::ClientAddressInvalid);
        }

        println!("{:?}", request);
        Ok(())
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

impl Dispatch<wl_compositor::WlCompositor, ()> for State {
    fn event(
        _: &mut Self,
        _: &wl_compositor::WlCompositor,
        _: wl_compositor::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // wl_compositor has no event
    }
}

impl Dispatch<wl_surface::WlSurface, ()> for State {
    fn event(
        _: &mut Self,
        _: &wl_surface::WlSurface,
        _: wl_surface::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // we ignore wl_surface events in this example
    }
}

impl Dispatch<wl_shm::WlShm, ()> for State {
    fn event(
        _: &mut Self,
        _: &wl_shm::WlShm,
        _: wl_shm::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // we ignore wl_shm events in this example
    }
}

impl Dispatch<wl_shm_pool::WlShmPool, ()> for State {
    fn event(
        _: &mut Self,
        _: &wl_shm_pool::WlShmPool,
        _: wl_shm_pool::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // we ignore wl_shm_pool events in this example
    }
}

impl Dispatch<wl_buffer::WlBuffer, ()> for State {
    fn event(
        _: &mut Self,
        _: &wl_buffer::WlBuffer,
        _: wl_buffer::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // we ignore wl_buffer events in this example
    }
}

impl Dispatch<xdg_wm_base::XdgWmBase, ()> for State {
    fn event(
        _: &mut Self,
        wm_base: &xdg_wm_base::XdgWmBase,
        event: xdg_wm_base::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_wm_base::Event::Ping { serial } = event {
            wm_base.pong(serial);
        }
    }
}

impl Dispatch<xdg_surface::XdgSurface, ()> for State {
    fn event(
        state: &mut Self,
        xdg_surface: &xdg_surface::XdgSurface,
        event: xdg_surface::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_surface::Event::Configure { serial, .. } = event {
            xdg_surface.ack_configure(serial);
            state.configured = true;
            let surface = state.base_surface.as_ref().unwrap();
            if let Some(ref buffer) = state.buffer {
                surface.attach(Some(buffer), 0, 0);
                surface.commit();
            }
        }
    }
}

impl Dispatch<xdg_toplevel::XdgToplevel, ()> for State {
    fn event(
        state: &mut Self,
        _: &xdg_toplevel::XdgToplevel,
        event: xdg_toplevel::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_toplevel::Event::Close {} = event {
            state.running = false;
        }
    }
}

// Implement `Dispatch<WlRegistry, ()> for out state. This provides the logic
// to be able to process events for the wl_registry interface.
//
// The second type parameter is the user-data of our implementation. It is a
// mechanism that allows you to associate a value to each particular Wayland
// object, and allow different dispatching logic depending on the type of the
// associated value.
//
// In this example, we just use () as we don't have any value to associate. See
// the `Dispatch` documentation for more details about this.
impl Dispatch<wl_registry::WlRegistry, ()> for State {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<State>,
    ) {
        // When receiving events from the wl_registry, we are only interested in the
        // `global` event, which signals a new available global.
        // When receiving this event, we just print its characteristics in this example.
        if let wl_registry::Event::Global {
            name, interface, ..
        } = event
        {
            match &interface[..] {
                "wl_seat" => {
                    registry.bind::<wl_seat::WlSeat, _, _>(name, 1, qh, ());
                }
                "wl_compositor" => {
                    let compositor =
                        registry.bind::<wl_compositor::WlCompositor, _, _>(name, 1, qh, ());
                    let surface = compositor.create_surface(qh, ());
                    state.base_surface = Some(surface);

                    if state.wm_base.is_some() && state.xdg_surface.is_none() {
                        state.init_xdg_surface(qh);
                    }
                }
                "wl_shm" => {
                    let shm = registry.bind::<wl_shm::WlShm, _, _>(name, 1, qh, ());

                    let (init_w, init_h) = (320, 240);

                    let mut file = tempfile::tempfile().unwrap();
                    draw(&mut file, (init_w, init_h));
                    let pool =
                        shm.create_pool(file.as_raw_fd(), (init_w * init_h * 4) as i32, qh, ());
                    let buffer = pool.create_buffer(
                        0,
                        init_w as i32,
                        init_h as i32,
                        (init_w * 4) as i32,
                        wl_shm::Format::Argb8888,
                        qh,
                        (),
                    );
                    state.buffer = Some(buffer.clone());

                    if state.configured {
                        let surface = state.base_surface.as_ref().unwrap();
                        surface.attach(Some(&buffer), 0, 0);
                        surface.commit();
                    }
                }
                "xdg_wm_base" => {
                    let wm_base = registry.bind::<xdg_wm_base::XdgWmBase, _, _>(name, 1, qh, ());
                    state.wm_base = Some(wm_base);

                    if state.base_surface.is_some() && state.xdg_surface.is_none() {
                        state.init_xdg_surface(qh);
                    }
                }
                _ => {}
            }
        }
    }
}

impl Dispatch<wl_seat::WlSeat, ()> for State {
    fn event(
        _: &mut Self,
        seat: &wl_seat::WlSeat,
        event: wl_seat::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_seat::Event::Capabilities {
            capabilities: WEnum::Value(capabilities),
        } = event
        {
            if capabilities.contains(wl_seat::Capability::Keyboard) {
                seat.get_keyboard(qh, ());
            }
        }
    }
}

impl Dispatch<wl_keyboard::WlKeyboard, ()> for State {
    fn event(
        _: &mut Self,
        _: &wl_keyboard::WlKeyboard,
        event: wl_keyboard::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wl_keyboard::Event::Key { state, .. } = event {
            match state {
                WEnum::Value(key_state) => {
                    let mut register_state = GLOBAL_REGISTERS.write().unwrap();
                    match key_state {
                        KeyState::Pressed => register_state.c = true,
                        KeyState::Released => register_state.c = false,
                        _ => panic!("Unknown keystate received!"),
                    }

                    println!("{:?}", &register_state);
                }
                WEnum::Unknown(value) => panic!("Unknown KeyState event {}", value),
            }
        }
    }
}

impl State {
    fn init_xdg_surface(&mut self, qh: &QueueHandle<State>) {
        let wm_base = self.wm_base.as_ref().unwrap();
        let base_surface = self.base_surface.as_ref().unwrap();

        let xdg_surface = wm_base.get_xdg_surface(base_surface, qh, ());
        let toplevel = xdg_surface.get_toplevel(qh, ());
        toplevel.set_title("Nunchuk Emulator".into());

        base_surface.commit();

        self.xdg_surface = Some((xdg_surface, toplevel));
    }
}

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

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct I2cArgs {
    /// Location of vhost-user Unix domain socket. This is suffixed by 0,1,2..socket_count-1.
    #[clap(short, long)]
    socket_path: String,

    /// Number of guests (sockets) to connect to.
    #[clap(short = 'c', long, default_value_t = 1)]
    socket_count: usize,

    /// List of I2C bus and clients in format
    /// <bus-name>:<client_addr>[:<client_addr>][,<bus-name>:<client_addr>[:<client_addr>]].
    #[clap(short = 'l', long)]
    device_list: String,
}

#[derive(Debug, PartialEq)]
struct DeviceConfig {
    adapter_name: String,
    addr: Vec<u16>,
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
    inner: Vec<DeviceConfig>,
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

fn start_backend<D>(args: I2cArgs) -> Result<(), EmulatorError>
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

    // for handle in handles {
    //     handle
    //         .join()
    //         .map_err(|_| EmulatorError::FailedJoiningThreads)?;
    // }

    Ok(())
}

// The main function of our program
fn main() -> anyhow::Result<()> {
    env_logger::init();

    // Create a Wayland connection by connecting to the server through the
    // environment-provided configuration.
    let conn = Connection::connect_to_env().unwrap();

    // Retrieve the WlDisplay Wayland object from the connection. This object is
    // the starting point of any Wayland program, from which all other objects will
    // be created.
    let display = conn.display();

    // Create an event queue for our event processing
    let mut event_queue = conn.new_event_queue();
    // An get its handle to associated new objects to it
    let qh = event_queue.handle();

    // Create a wl_registry object by sending the wl_display.get_registry request
    // This method takes two arguments: a handle to the queue the newly created
    // wl_registry will be assigned to, and the user-data that should be associated
    // with this registry (here it is () as we don't need user-data).
    let _registry = display.get_registry(&qh, ());

    // At this point everything is ready, and we just need to wait to receive the events
    // from the wl_registry, our callback will print the advertized globals.
    println!("Advertized globals:");

    // To actually receive the events, we invoke the `sync_roundtrip` method. This method
    // is special and you will generally only invoke it during the setup of your program:
    // it will block until the server has received and processed all the messages you've
    // sent up to now.
    //
    // In our case, that means it'll block until the server has received our
    // wl_display.get_registry request, and as a reaction has sent us a batch of
    // wl_registry.global events.
    //
    // `sync_roundtrip` will then empty the internal buffer of the queue it has been invoked
    // on, and thus invoke our `Dispatch` implementation that prints the list of advertized
    // globals.
    let mut state = State {
        running: true,
        ..Default::default()
    };

    start_backend::<Nunchuk>(I2cArgs::parse())?;
    while state.running {
        event_queue.blocking_dispatch(&mut state).unwrap();
    }

    Ok(())
}
