# TransformerScope
An interpretability tool for transformer models built on TransformerLens

# Contributor setup
This guide will ensure you have the right environment and run the [othelloscope example](examples/othelloscope)
1. Ensure you have a working Python installation
2. Ensure you have a working Rust toolchain. See [here](https://www.rust-lang.org/tools/install) to get one.
3. Clone the repo and move to the root.
4. Set up a python environment using `python -m venv .env` and activate it.
5. Install [Maturin](https://github.com/PyO3/maturin) with `python -m pip install maturin`
6. Build the package by moving to the `transformer-scope` directory and running `python -m maturin develop --release`. The package will now be installed in your environment
7. Move back to the root and run `python -m pip install -r examples/othelloscope/requirements.txt` to install the example's dependencies.
8. Run `python -m examples.othelloscope.main` and wait as the example runs.
9. Run `cargo run --release -- examples/othelloscope/output/payload` to start the server.
10. Visit [localhost:8080](localhost:8080) in a browser to view the website.
