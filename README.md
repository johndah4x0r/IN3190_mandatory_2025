# IN3190_mandatory_2025
Mandatory project for the IN3190 DSP course

## Usage
### Using `make`
If possible, use the provided `Makefile`, and run
```bash
make purge && make PROJECT_DATA_ZIP=<path-to-archive>
```

### Using `./run_all.py`
Assuming that `.h5` files with seismic data are stored in `./seismic_data/`,
the run script can be executed directly:
```bash
python3 ./run_all.py
```
