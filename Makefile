.DEFAULTGOAL := all

# Project data archive (must be provided explicitly)
# - dummy value provided
PROJECT_DATA_ZIP := archives/project_data.zip

all: seismic_data/
	python3 run_all.py

clean:
	rm -f unpacked.h5 task_*.npz
	rm -rf tmpfigs

purge: clean
	rm -rf seismic_data/

seismic_data/:
	unzip $(PROJECT_DATA_ZIP) -d $@

.PHONY: all clean purge