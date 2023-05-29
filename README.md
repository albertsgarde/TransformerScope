# TransformerScope
An interpretability tool for transformer models built on [TransformerLens](https://github.com/neelnanda-io/TransformerLens)

## Contributor setup
This guide will ensure you have the right environment and run the [othelloscope example](examples/othelloscope).
Tested in Windows Subsystem for Linux with Ubuntu 22.04.2 LTS.
Also see the [`environment.yml`](.github/workflows/environment.yml) workflow for a CI tested setup procedure.
1. Ensure you have a working Python installation (at least version 3.7, tested with version 3.10.7).
2. Ensure you have a working Rust toolchain (if you can use the `cargo` command it should be fine). 
   See [here](https://www.rust-lang.org/tools/install) to get one.
   Any version from the last few years should work.
   The newest one definitely will.
3. Clone the repo and move to the root.
4. Set up a python environment using `python -m venv .env`.
5. Activate the environment.
6. Install globally required packages by running `python -m pip install -r requirements.txt`.
7. Build the package by moving to the `transformer-scope` directory and running `maturin develop --release`.
   The package will now be installed in your environment.
7. Move back to the root and run `python -m pip install -r examples/othelloscope/requirements.txt` to install the example's dependencies.
8. Run `python -m examples.othelloscope.main` and wait as the example runs.
9. Run `cargo run --release -- examples/othelloscope/output/payload` to start the server.
10. Wait until it displays the message `Serving site...`.
11. Visit [localhost:8080](localhost:8080) in a browser to view the website.

### Windows notes
On Windows, Maturin works less well, but there are work arounds.
1. Make sure you clone the project into a path with no spaces.
2. When building with Maturin, if you get the error `Invalid python interpreter version` or `Unsupported Python interpreter`, this is likely because Maturin fails to find your environment's interpreter.
To fix this, instead of building with `maturin develop`, use `maturin build --release -i py.exe` (maybe replace `py.exe` with e.g. `python3.exe` if that is how you call Python) and then call `python -m pip install .`.
The `-i` argument tells Maturin the name of the Python interpreter to use.

### M1 notes
Problems arise when your Python version does not match your machines architecture.
This can happen on M1 chips, since it is possible to run x86 Python even if the architecture is ARM.
In this case you can get an error that looks like 
```
error[E0463]: can't find crate for `core`
  |
  = note: the `x86_64-apple-darwin` target may not be installed
  = help: consider downloading the target with `rustup target add x86_64-apple-darwin
``` 
Simply download the x86 target with the suggested command and everything should work.
