# C3DP Automated Mixture Discovery Workflow

This repository contains the codebase for the paper Discovery of 3D-Printed Construction Materials via Intelligent Data Extraction and Generation using Large Language Models.

## Repository Functionality and Layout

### Data Extraction and Expansion

The subroutine code for data extraction is located in `./tools/extractor.py`. It accesses prompts in the `./prompts` directory to read papers, extract strength and material composition data, correct percentages, and postprocess the extracted dataset. The subroutine code for data expansion is located in `./tools/refscrape.py`. It reads through the provided papers and suggests additional papers to process for C3DP strength data extraction.

### Strength Learning

The subroutine code for strength learning is located in `./tools/matpred.py`. It preprocesses the extracted C3DP strength data before training Random Forest models and saving them in the `./save/` directory.

### Material Discovery

The subroutine code for material discovery is located in `./tools/discover.py`. It uses the proposed optimization procedure to adjust existing samples towards novel C3DP mixtures that maximize strength and minimize cost and carbon footprint. Resulting discovered samples are stored in `./db/discover_samples.csv`.

## Data

The officially generated LLM-Generated Database is in `./db/lgd_db.npy`. The official benchmark Manual Curated Database is in `./db/mcd_db.csv`.

## Usage

First, ensure that the papers you wish to extract data from are in the `./db/papers/` directory. You should also populate your Gemini 2.5 API key in the relevant subroutine files in the `./tools/` director (search for "TODO"). Data extraction can be performed using the following command:

`python main.py --opt = 1`

After data expansion has completed, data expansion can be performed. You should populate the `./util/starter_auth.txt` file with at least one last name for starting authors to expand from. You can add to this list, and the list in `./util/starter_ven.txt` for starting venues, to further expand the acceptable space for papers. Data expansion can be performed using the following command:

`python main.py --opt = 2`

After data expansion has completed, additional data can be extracted. Move the newly suggested papers to the `./db/papers/` directory and rerun the above data extraction command.

Strength learning can be performed using the following command:

`python main.py --opt = 3`

Material discovery can be performed using the following command:

`python main.py --opt = 4`

Note that the subroutines are designed to be run sequentially. They cannot be run independently.

## Acknowledgments

TODO
