Description of the dataset
==========================
The Crisis Benchmarks Dataset consists of data from different data sources [1]):


Data format and directories
===========================
The data directory contains the following sub-directories. In each directory, there are files for Informativeness and Humanitarian tasks.

* data/all_data_en -- all combined english dataset used for the experiments
* data/individual_data_en -- consists of data used for the experiments as individual data source such as crisisnlp and crisislex
* data/event_aware_en -- all combined english dataset with event tag (fire, earthquake, flood, ...) are tagged
* data/data_split_all_lang -- all combined dataset with their train/dev and test splits
* data/initial_filtering -- all combined dataset duplicate removed data
* data/class_label_mapped -- all combined dataset initial set of dataset where class label mapped

Format of the TSV files
---------------------------------------------------------
Each TSV file contains the following columns, separated by a tab:
* id: corresponds to the user tweet id from Twitter.
* event: event name associated with the respective dataset
* source: source of the dataset
* text: tweet text.
* lang: language tag obtained either from Twitter or obtained from Google language detection API
* lang conf: confidence score obtained from Google language detection API; for many cases, there are tag "NA", which represents that the language tag is obtained from Twitter from Google API
* class_label: class label a given tweet text.


Citation
========
If you use this data in your research, please consider citing the following paper:
[1] Firoj Alam, Hassan Sajjad, Muhammad Imran and Ferda Ofli, CrisisBench: Benchmarking Crisis-related Social Media Datasets for Humanitarian Information Processing, In ICWSM, 2021.


@inproceedings{firojalamcrisisbenchmark2020,
Author = {Firoj Alam, Hassan Sajjad, Muhammad Imran, Ferda Ofli},
Keywords = {Social Media, Crisis Computing, Tweet Text Classification, Disaster Response},
Booktitle = {15th International Conference on Web and Social Media (ICWSM)},
Title = {CrisisBench: Benchmarking Crisis-related Social Media Datasets for Humanitarian Information Processing},
Year = {2021}
}

* and the following associated papers

* Muhammad Imran, Prasenjit Mitra, Carlos Castillo. Twitter as a Lifeline: Human-annotated Twitter Corpora for NLP of Crisis-related Messages. In Proceedings of the 10th Language Resources and Evaluation Conference (LREC), 2016, Slovenia.
* A. Olteanu, S. Vieweg, C. Castillo. 2015. What to Expect When the Unexpected Happens: Social Media Communications Across Crises. In Proceedings of the ACM 2015 Conference on Computer Supported Cooperative Work and Social Computing (CSCW '15). ACM, Vancouver, BC, Canada.
* A. Olteanu, C. Castillo, F. Diaz, S. Vieweg. 2014. CrisisLex: A Lexicon for Collecting and Filtering Microblogged Communications in Crises. In Proceedings of the AAAI Conference on Weblogs and Social Media (ICWSM'14). AAAI Press, Ann Arbor, MI, USA.
* Muhammad Imran, Shady Elbassuoni, Carlos Castillo, Fernando Diaz and Patrick Meier. Practical Extraction of Disaster-Relevant Information from Social Media. In Social Web for Disaster Management (SWDM'13) - Co-located with WWW, May 2013, Rio de Janeiro, Brazil.
* Muhammad Imran, Shady Elbassuoni, Carlos Castillo, Fernando Diaz and Patrick Meier. Extracting Information Nuggets from Disaster-Related Messages in Social Media. In Proceedings of the 10th International Conference on Information Systems for Crisis Response and Management (ISCRAM), May 2013, Baden-Baden, Germany.


@inproceedings{imran2016lrec,
  author = {Muhammad Imran and Prasenjit Mitra and Carlos Castillo},
  title = {Twitter as a Lifeline: Human-annotated Twitter Corpora for NLP of Crisis-related Messages},
  booktitle = {Proc. of the LREC, 2016},
  year = {2016},
  month = {5},
  publisher = {ELRA},
  address = {Paris, France},
  isbn = {978-2-9517408-9-1},
  language = {english}
 }
 @inproceedings{olteanu2015expect,
  title={What to expect when the unexpected happens: Social media communications across crises},
  author={Olteanu, Alexandra and Vieweg, Sarah and Castillo, Carlos},
  booktitle={Proc. of the 18th ACM Conference on Computer Supported Cooperative Work \& Social Computing},
  pages={994--1009},
  year={2015},
  organization={ACM}
}
@inproceedings{olteanu2014crisislex,
  title={CrisisLex: A Lexicon for Collecting and Filtering Microblogged Communications in Crises.},
  author={Olteanu, Alexandra and Castillo, Carlos and Diaz, Fernando and Vieweg, Sarah},
  booktitle = "Proc. of the 8th ICWSM, 2014",
  publisher = "AAAI press",
  year={2014}
}
@inproceedings{imran2013practical,
  title={Practical extraction of disaster-relevant information from social media},
  author={Imran, Muhammad and Elbassuoni, Shady and Castillo, Carlos and Diaz, Fernando and Meier, Patrick},
  booktitle={Proc. of the 22nd WWW},
  pages={1021--1024},
  year={2013},
  organization={ACM}
}
@inproceedings{imran2013extracting,
  title={Extracting information nuggets from disaster-related messages in social media},
  author={Imran, Muhammad and Elbassuoni, Shady Mamoon and Castillo, Carlos and Diaz, Fernando and Meier, Patrick},
  booktitle={Proc. of the 12th ISCRAM},
  year={2013}
}

Terms of Use
============
Please follow the terms of use mentioned here:
https://crisisnlp.qcri.org/terms-of-use.html
