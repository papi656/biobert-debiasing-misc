echo "Running bias product"
echo "Model - early stopped"
echo "Dataset - BC5CDR"
echo ""

python biobert_clf_debias.py \
    --dataset_name BC5CDR \
    --loss_fn BiasProduct \
    --bias_file BC5CDR_prior_prob.txt \
    --model_name model 

echo "Generating prediction file using debiased-early stopped model"

python inference.py \
    --dataset_name BC5CDR \
    --model_name model_BiasProduct \
    --loss_fn BiasProduct

echo ""
echo "Performance on BC5CDR using debiased-early stopped model"
echo ""

python evaluate.py \
    --mention_dictionary datasets/BC5CDR/mention_dict.txt \
    --cui_dictionary datasets/BC5CDR/cui_dict.txt \
    --gold_labels datasets/BC5CDR/test.txt \
    --gold_cuis datasets/BC5CDR/test_cuis.txt \
    --predictions resources/BC5CDR/preds_model_BiasProduct.txt 

echo ""
echo "Running bias product"
echo "Model - 30 epochs trained"
echo "Dataset - BC5CDR"
echo ""

python biobert_clf_debias.py \
    --dataset_name BC5CDR \
    --loss_fn BiasProduct \
    --bias_file BC5CDR_prior_prob.txt \
    --model_name model_30epochs

echo "Generating prediction file using debiased-30 epochs model"

python inference.py \
    --dataset_name BC5CDR \
    --model_name model_30epochs_BiasProduct \
    --loss_fn BiasProduct

echo ""
echo "Performance on BC5CDR using debiased-30 epochs model"
echo ""

python evaluate.py \
    --mention_dictionary datasets/BC5CDR/mention_dict.txt \
    --cui_dictionary datasets/BC5CDR/cui_dict.txt \
    --gold_labels datasets/BC5CDR/test.txt \
    --gold_cuis datasets/BC5CDR/test_cuis.txt \
    --predictions resources/BC5CDR/preds_model_30epochs_BiasProduct.txt 


