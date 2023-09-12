TO REPRODUCE THE EXPERIMENT DESCRIBED IN THE PAPER:

1. Install the dependencies as per requirements.txt (in particular, torch==1.11.0)
2. Download the glove.840B.300d word embeddings from https://nlp.stanford.edu/projects/glove/
3. The data is provided under "data". It was extracted from the ERG treebanks 2020 release (see link in the paper).
4. Now simply run the (slightly modified) NCRF++ on the data:
    python3 main.py --config erg.tran-best.config
5. To decode, run:
    python3 main.py --config erg.decode.config

You can change the test data in the config to obtain numbers for each test corpus separately.

FOR THE BASELINE MODEL:
1. Prepare the data:
   1.1. Download the ERG treebanks from svn checkout http://svn.delph-in.net/erg/tags/2020
   1.2. Locate the treebanks under tsdb/gold/
   1.3. Prepare the data split as recommended in the file redwoods.xls (found in the same ERG archive). Put all the training corpora in a folder called "train", all the dev ones in "dev", and all the test ones in "test". You should have 57 corpora in train, 2 in dev, and 14 in test.

2. Put the ERG lexicons in a separate folder, you should have 4 files there: gle.tdl, lexicon-rbst.tdl, lexicon.tdl, and ple.tdl 

3. Run letype_extractor as follows: 

    python3 letype_extractor.py path-to-lexicons/ path-to-treebanks/ custom-name-for-run- nonautoreg

(Just use any name for your run, such as "paper-repro" or whatever. The "nonautoreg" flag is needed in some cases to train autoregressive vs nonautoregressive models; just put "nonautoreg" there.
This step will produce a folder under "output/your-run-name" with data in it.

4. Now run the vectorizer: 
   python3 vectorizer.py path-to-your-run-folder/ nonautoreg
 
 This will create the vectorized data under the same folder.
 
 5. Now train the models (this will train two models: SVM and MaxEnt OVR L1 SAGA: 
 
   python3 classic_classifiers.py train  output/your-run-name/ nonautoreg
   
   (the OVR MaxEnt model may take up to 48 hours to train, depending on the machine. SVM trains under a few hours).
   
 5. To decode, for dev or test:
 
    python3 classic_classifiers.py test output/your-run-name/ dev nonautoreg
    
    or:
    
    python3 classic_classifiers.py test output/your-run-name/ test nonautoreg
   
 
