use std::env;

use actix_web::{get, web, App, HttpServer, Responder};
use transformer_scope::{html, ApplicationState};

#[get("/")]
async fn index(data: web::Data<ApplicationState>) -> impl Responder {
    let data = data.as_ref();
    html::index_html(data)
}

#[get("/L{layer_index}/N{neuron_index}")]
async fn neuron(
    data: web::Data<ApplicationState>,
    path: web::Path<(usize, usize)>,
) -> impl Responder {
    let (layer_index, neuron_index) = path.into_inner();

    html::neuron_html(layer_index, neuron_index, data.as_ref())
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let path = env::args().nth(1).unwrap();

    let state = ApplicationState::new(path);

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(state.clone()))
            .service(index)
            .service(neuron)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
