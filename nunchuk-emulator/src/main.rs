use std::cell::RefCell;
use std::env;

use backend::{Button, ButtonEvent, ButtonState, I2cArgs, Nunchuk};
use gtk::prelude::*;
use gtk::subclass::prelude::*;
use gtk::{
    gdk::{Key, Texture},
    glib::{
        self,
        subclass::{object::ObjectImpl, types::ObjectSubclass},
        Propagation,
    },
    ApplicationWindow, EventControllerKey, Image,
};

mod backend;
mod i2c;
mod vhu_i2c;

const APP_ID: &str = "com.ethantwardy.NunchukEmulator";

fn key_handler(state: ButtonState, keyval: Key) {
    let Some(keycode) = keyval.to_unicode() else {
        return;
    };

    let event = match keycode {
        'c' => ButtonEvent {
            button: Button::C,
            state,
        },
        'z' => ButtonEvent {
            button: Button::Z,
            state,
        },
        _ => return,
    };

    backend::update_buttons(event);
}

mod wrapper {
    use gtk::{gio, glib};

    glib::wrapper! {
        pub struct NunchukEmulator(ObjectSubclass<super::NunchukEmulator>)
            @extends gio::Application, gtk::Application;
    }

    impl Default for NunchukEmulator {
        fn default() -> Self {
            glib::Object::builder()
                .property("application-id", super::APP_ID)
                .property("nunchuk-type", super::NUNCHUK_WHITE)
                .build()
        }
    }
}

const NUNCHUK_WHITE: i32 = 0;

#[derive(glib::Properties)]
#[properties(wrapper_type = wrapper::NunchukEmulator)]
pub struct NunchukEmulator {
    #[property(get, set)]
    nunchuk_type: RefCell<i32>,
}

impl Default for NunchukEmulator {
    fn default() -> Self {
        Self {
            nunchuk_type: RefCell::new(NUNCHUK_WHITE),
        }
    }
}

#[glib::object_subclass]
impl ObjectSubclass for NunchukEmulator {
    const NAME: &'static str = "NunchukEmulator";
    type Type = wrapper::NunchukEmulator;
    type ParentType = gtk::Application;
}

#[glib::derived_properties]
impl ObjectImpl for NunchukEmulator {}

impl GtkApplicationImpl for NunchukEmulator {}
impl ApplicationImpl for NunchukEmulator {
    fn activate(&self) {
        let nunchuk_white = include_bytes!("../../images/nunchuk-white.jpg");
        let image = nunchuk_white.as_slice();

        let texture = Texture::from_bytes(&image.into()).ok();
        let image = Image::from_paintable(texture.as_ref());

        let key_events = EventControllerKey::builder().build();
        key_events.connect_key_pressed(|_, keyval, _, _| {
            key_handler(ButtonState::Pressed, keyval);
            Propagation::Stop
        });
        key_events.connect_key_released(|_, keyval, _, _| {
            key_handler(ButtonState::Unpressed, keyval);
        });

        let window = ApplicationWindow::builder()
            .application(&*self.obj())
            .title("Nunchuk Emulator")
            .child(&image)
            .build();

        window.add_controller(key_events);
        window.present();
    }
}

fn main() -> glib::ExitCode {
    let xdg_runtime_dir = env::var("XDG_RUNTIME_DIR").unwrap();
    let socket_path = format!("{xdg_runtime_dir}/vhost-user-i2c-1.sock");
    let args = I2cArgs {
        socket_path,
        socket_count: 1,
        // NOTE: 82d == 52h
        device_list: "i2c-1:82".to_string(),
    };
    backend::start_backend::<Nunchuk>(args).unwrap();
    let app = wrapper::NunchukEmulator::default();
    app.run()
}
