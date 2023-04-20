use std::fs::File;

use actix_test::board_heatmap;
use actix_web::{get, web, App, HttpResponse, HttpServer, Responder};
use ndarray::{s, Array4};
use ndarray_npy::ReadNpyExt;

#[get("/")]
async fn hello() -> impl Responder {
    println!("helleflynder");
    HttpResponse::Ok().body("Hello world!")
}

#[get("/L{layer_index}/N{neuron_index}")]
async fn echo(path: web::Path<(usize, usize)>) -> impl Responder {
    let (layer_index, neuron_index) = path.into_inner();

    let path = "heatmaps_my.npy";
    let file = File::open(path).unwrap();
    let array = Array4::<f32>::read_npy(file).unwrap();

    let values = array
        .slice(s![layer_index, neuron_index, .., ..])
        .to_owned();

    board_heatmap::board_heatmap(&values)
}

async fn manual_hello() -> impl Responder {
    HttpResponse::Ok().body("Hey there!")
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .service(hello)
            .service(echo)
            .route("/hey", web::get().to(manual_hello))
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
