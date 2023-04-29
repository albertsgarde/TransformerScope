use std::{
    env,
    io::{self, Write},
};

use actix_web::{get, web, App, HttpServer, Responder};
use transformer_scope::html;

mod state;
use state::ApplicationState;

#[get("/")]
async fn index(data: web::Data<ApplicationState>) -> impl Responder {
    let data = data.as_ref();
    html::generate_index_page(data.payload())
}

#[get("/L{layer_index}/N{neuron_index}")]
async fn neuron(
    data: web::Data<ApplicationState>,
    path: web::Path<(usize, usize)>,
) -> impl Responder {
    let (layer_index, neuron_index) = path.into_inner();

    html::generate_neuron_page(layer_index, neuron_index, data.payload())
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let path = env::args().nth(1).unwrap();

    print!("Loading payload...");
    io::stdout().flush().unwrap();
    let state = ApplicationState::new(path);
    println!("\rPayload loaded.                  ");

    HttpServer::new(move || {
        App::new()
            .service(actix_files::Files::new("/static", "./static"))
            .app_data(web::Data::new(state.clone()))
            .service(index)
            .service(neuron)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
