from task_1 import movies_reviews
from task_2 import model_trainer
from task_3 import translate_blue


if __name__ == '__main__':
    print("Task 1: Out of the Box Sentiment Analysis ")
    task_1 = movies_reviews()
    task_1.main()
    print("-----------END------------")
    print("Task 2: Take a basic, pretrained NER model, and train further on a task-specific dataset")
    samples_train = 300
    samples_test = 30
    task_2 = model_trainer(samples_test, samples_train)
    task_2.main()
    task_2.graphic_model()
    print("-----------END------------")
    print("Task 3:Set up and compare model performance of two different translation models")
    lang1_set = 'en_corpus.txt'
    lang2_set = 'es_corpus.txt'
    lang_from = 'en'
    lang_to = 'es'
    cod_key = 'c72e1edd13a1410785bef1c40dd6224e'
    cod_region = 'southcentralus'
    gcp_keys = 'private_key.json'
    task_3 = translate_blue(lang1_set, lang2_set, lang_from, lang_to, cod_key, cod_region, gcp_keys)
    task_3.main()
    print("-----------END------------")
