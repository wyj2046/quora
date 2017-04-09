gcloud ml-engine local train \
       --package-path ./trainer \
       --module-name trainer.deepnet1 \
       -- \
       --gcp \
       --param_config nn1.conf \
       --data_path gs://$BUCKET_NAME/data/all_data.pickle
