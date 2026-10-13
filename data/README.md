# Data

The occupancy series (`bed_occupancy1.csv`: columns `DATE` in `%d-%b-%y` format and `BED_OCCUPANCY`, 3,378 daily
values from 1 January 2011 to 31 March 2020) belongs to the institute and is not distributed here. It can be made
available on formal request, through a secure channel and subject to institutional permission; see the
corresponding author's contact details in the paper.

The scripts expect the file at `data/bed_occupancy1.csv` and check its SHA-256 checksum before use
(`pipeline/common.py`).
