use std::env;

use actix_test::{board_heatmap, ApplicationState};
use actix_web::{get, web, App, HttpResponse, HttpServer, Responder};
use ndarray::s;

#[get("/")]
async fn hello() -> impl Responder {
    println!("helleflynder");
    HttpResponse::Ok().body("Hello world!")
}

#[get("/L{layer_index}/N{neuron_index}")]
async fn echo(
    data: web::Data<ApplicationState>,
    path: web::Path<(usize, usize)>,
) -> impl Responder {
    let (layer_index, neuron_index) = path.into_inner();

    let ownership_heatmaps = data.ownership_heatmaps();

    let values = ownership_heatmaps
        .slice(s![layer_index, neuron_index, .., ..])
        .to_owned();

    board_heatmap::board_heatmap(&values)
}

async fn manual_hello() -> impl Responder {
    HttpResponse::Ok().body("Hey there!")
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let path = env::args().nth(1).unwrap();

    let state = ApplicationState::new(path);

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(state.clone()))
            .service(hello)
            .service(echo)
            .route("/hey", web::get().to(manual_hello))
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
