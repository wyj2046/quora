gcloud ml-engine local train \
       --package-path ./trainer \
       --module-name trainer.deepnet2 \
       -- \
       --gcp \
       --param_config nn2.conf \
       --data_path gs://$BUCKET_NAME/data/all_data.pickle
