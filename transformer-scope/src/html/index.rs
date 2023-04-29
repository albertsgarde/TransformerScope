use maud::{html, Markup};

use crate::ApplicationState;

pub fn index_html(_state: &ApplicationState) -> Markup {
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
