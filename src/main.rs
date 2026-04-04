mod api;
mod app;
mod message;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([700.0, 500.0])
            .with_title("hChat"),
        ..Default::default()
    };

    eframe::run_native("hChat", options, Box::new(|cc| Ok(Box::new(app::ChatApp::new(cc)))))
}
