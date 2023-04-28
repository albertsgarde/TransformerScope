use std::env;

use actix_web::{get, web, App, HttpServer, Responder};
use maud::html;
use ndarray::s;
use transformer_scope::{board_heatmap, ApplicationState};

#[get("/")]
async fn index() -> impl Responder {
    html! {
        head {
            meta charset="utf-8";
            title { "Transformer Scope" }
        }
        body {
            h1 { "Transformer Scope" }
            p { "This is a web app for visualizing various aspects of transformer models." }
            p { "This page should include an index, but this has not been implemented yet." }
        }
    }
}

#[get("/L{layer_index}/N{neuron_index}")]
async fn neuron(
    data: web::Data<ApplicationState>,
    path: web::Path<(usize, usize)>,
) -> impl Responder {
    let (layer_index, neuron_index) = path.into_inner();

    let ownership_heatmaps = data.ownership_heatmaps();

    let values = ownership_heatmaps
        .slice(s![layer_index, neuron_index, .., ..])
        .to_owned();

    let heatmap_html = board_heatmap::board_heatmap(&values);
    html! {
        head {
            meta charset="utf-8";
            title { "Transformer Scope - Layer " (layer_index) " Neuron " (neuron_index)}
        }
        (heatmap_html)
    }
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
