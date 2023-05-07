use itertools::Itertools;
use maud::{html, Markup, PreEscaped, DOCTYPE};
use serde::{Deserialize, Serialize};

use super::Element;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronTemplate {
    prefix: String,
    template: Vec<(Element, String)>,
}

impl NeuronTemplate {
    fn parse_inner(template_code: &str) -> Self {
        let mut parts = template_code.split('$');
        let prefix = parts.next().unwrap().to_string();
        let template: Vec<_> = parts
            .map(|s| {
                let mut parts = s.split_inclusive(')');
                let element_string = parts.next().unwrap();
                let template_part = parts.join("");
                let element = Element::parse(element_string);
                (element, template_part)
            })
            .collect();
        Self { prefix, template }
    }

    pub fn parse<S: AsRef<str>>(template_code: S) -> Self {
        Self::parse_inner(template_code.as_ref())
    }

    pub fn generate(
        &self,
        payload: &crate::Payload,
        file: bool,
        layer_index: usize,
        neuron_index: usize,
    ) -> Markup {
        let mut body = self.prefix.to_string();
        for (element, template_part) in &self.template {
            let element_markup = element.generate(payload, layer_index, neuron_index);
            body.push_str(&element_markup.into_string());
            body.push_str(template_part);
        }

        html!(
            (DOCTYPE)
            head {
                meta charset="utf-8";
                title { "Transformer Scope - Layer " (layer_index) " Neuron " (neuron_index)}
                link rel="stylesheet" href={(if file {".."} else {""})"/static/style.css"}{};
            }
            a href={(if file {"../index.html"} else {"/"})} {"Back to index"}
            h1 {"Transformer Scope - Layer " (layer_index) " Neuron " (neuron_index)}
            (generate_navigation_links(payload.num_layers(), payload.num_mlp_neurons(), layer_index, neuron_index, file))
            (PreEscaped(body))
        )
    }
}

fn generate_navigation_links(
    num_layers: usize,
    num_neurons: usize,
    layer_index: usize,
    neuron_index: usize,
    file: bool,
) -> Markup {
    let file_extension = if file { ".html" } else { "" };
    let previous_neuron_link = if neuron_index > 0 {
        html! {
            a href={"N"({neuron_index-1})(file_extension)} {
                "Previous"
            }
        }
    } else if layer_index > 0 {
        html! {
            a href={"../L"({layer_index-1})"/N"({num_neurons-1})(file_extension)} {
                "Previous layer"
            }
        }
    } else {
        html! {}
    };

    let next_neuron_link = if neuron_index < num_neurons - 1 {
        html! {
            a href={"N"({neuron_index+1})(file_extension)} {
                "Next"
            }
        }
    } else if layer_index < num_layers - 1 {
        html! {
            a href={"../L"({layer_index+1})"/N0"(file_extension)} {
                "Next layer"
            }
        }
    } else {
        html! {}
    };

    html! {
        (previous_neuron_link)" - "(next_neuron_link)
    }
}
