mod api;
mod app;
mod config;
mod message;
mod storage;

fn main() -> eframe::Result<()> {
    let config = config::Config::load();

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([900.0, 600.0])
            .with_title("hChat"),
        ..Default::default()
    };

    eframe::run_native(
        "hChat",
        options,
        Box::new(move |cc| Ok(Box::new(app::ChatApp::new(cc, config)))),
    )
}
