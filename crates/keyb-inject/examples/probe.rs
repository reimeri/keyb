//! Probe injection backends without sending any events: constructs each
//! backend and reports what the running session supports.
fn main() {
    let chars: Vec<char> = ('a'..='z').chain([' ', '.', ',', '\'']).collect();
    match keyb_inject::WaylandInjector::new(&chars) {
        Ok(i) => {
            use keyb_inject::Injector;
            println!("wayland: OK, caps = {:?}", i.capabilities());
        }
        Err(e) => println!("wayland: unavailable ({e:#})"),
    }
    match keyb_inject::XtestInjector::new() {
        Ok(i) => {
            use keyb_inject::Injector;
            println!("x11:     OK, caps = {:?}", i.capabilities());
        }
        Err(e) => println!("x11:     unavailable ({e:#})"),
    }
}
