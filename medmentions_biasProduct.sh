echo "Running bias product"
echo "Model - early stopped"
echo "Dataset - MedMentions"
echo ""

python biobert_clf_debias.py \
    --dataset_name MedMentions \
    --loss_fn BiasProduct \
    --bias_file MedMentions_prior_prob.txt \
    --model_name model 

echo "Generating prediction file using debiased-early stopped model"

python inference.py \
    --dataset_name MedMentions \
    --model_name model_BiasProduct \
    --loss_fn BiasProduct

echo ""
echo "Performance on MedMentions using debiased-early stopped model"
echo ""

python evaluate.py \
    --mention_dictionary datasets/MedMentions/mention_dict.txt \
    --cui_dictionary datasets/MedMentions/cui_dict.txt \
    --gold_labels datasets/MedMentions/test.txt \
    --gold_cuis datasets/MedMentions/test_cuis.txt \
    --predictions resources/MedMentions/preds_model_BiasProduct.txt 

echo ""
echo "Running bias product"
echo "Model - 30 epochs trained"
echo "Dataset - MedMentions"
echo ""

python biobert_clf_debias.py \
    --dataset_name MedMentions \
    --loss_fn BiasProduct \
    --bias_file MedMentions_prior_prob.txt \
    --model_name model_30epochs

echo "Generating prediction file using debiased-30 epochs model"

python inference.py \
    --dataset_name MedMentions \
    --model_name model_30epochs_BiasProduct \
    --loss_fn BiasProduct

echo ""
echo "Performance on MedMentions using debiased-30 epochs model"
echo ""

python evaluate.py \
    --mention_dictionary datasets/MedMentions/mention_dict.txt \
    --cui_dictionary datasets/MedMentions/cui_dict.txt \
    --gold_labels datasets/MedMentions/test.txt \
    --gold_cuis datasets/MedMentions/test_cuis.txt \
    --predictions resources/MedMentions/preds_model_30epochs_BiasProduct.txt 


