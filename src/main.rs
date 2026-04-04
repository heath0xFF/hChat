mod api;
mod app;
mod message;
mod storage;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([900.0, 600.0])
            .with_title("hChat"),
        ..Default::default()
    };

    eframe::run_native("hChat", options, Box::new(|cc| Ok(Box::new(app::ChatApp::new(cc)))))
}
