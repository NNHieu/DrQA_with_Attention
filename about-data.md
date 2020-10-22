The provided data consists of three files:

1. train.json: this is the training data which consists of ~18K annotated items. Each item includes:

* id : the tracking ID of the item
* question: the question
* text : the 'relevant' text of the question using a dummy search tool. It is a paragraph in a Wikipedia article.
* title: the title of the Wikipedia article from which the text has been collected.
* label : indicates whether or not the text can be an answer of the question.

 

2. test.json: this contains all test cases of in the public test set. Each test case includes:

* `__id__`: the test case ID
* `question`: the question
* `paragraphs`: extracted paragraphs using a dummy search tool. Each paragraph has a unique ID in a test case. The participating team must find IDs of paragraphs that can answer the question.
* `title`: the title of the Wikipedia articles from which paragraphs are selected.

    Note: Related paragraphs may be shuffled, a test case may have multiple answers or no answer.

 

3. sample_submission.csv: This is the sample for a submission. Headers of the table are the test_id and answer. 

The value of each row indicates the ID of the test case and paragraph that can answer the question in that test case. Thus, if a test case has `n` answers, there will `n` rows whose values at the `test_id` column are the same.

Besides, a test case which has no answer will not appear in the submission file. The F1 measure then is calculated by comparing the submitted answers with the gold answers. 

Table 1. Sample submission file. 

Explanation: test case test_0001 has 2 answers, but test_0003 has only one answer
test_id	answer
test_0001	p1
test_0001	p2
test_0003	p1

 

Download training file (updated on Oct 21, 2019): https://dl.challenge.zalo.ai/ZAC2019_VietnameseWikiQA/train.zip

Download test file: https://dl.challenge.zalo.ai/ZAC2019_VietnameseWikiQA/test.zip

Download sample submission file: https://dl.challenge.zalo.ai/ZAC2019_VietnameseWikiQA/sample_submmision.csv

 

4. Private test

    Download here: https://dl.challenge.zalo.ai/ZAC2019_VietnameseWikiQA/private_test/test.zip

 

Relevant Materials

On November 02 & 10, 2019, "Building Question Answering Systems" topic was shared and discussed in Zalo Tech Talk events. Download the presentation file here: https://dl.challenge.zalo.ai/ZAC2019_VietnameseWikiQA/ZAC2019_BuildingQAS_Slides.pdf