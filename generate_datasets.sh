#!/usr/bin/env bash

BKGDIR="./backgrounds"
EXAMPLEDIR="./examples"

# if $BKGDIR doesn't exist.
if [ ! -d "$BKGDIR" ]; then
    # download backgrounds
    wget https://www.dropbox.com/s/7qlfg1fywysz54q/backgrounds.tar.gz?dl=1 -O backgrounds.tar.gz
    tar -xzf backgrounds.tar.gz
fi

# if $BKGDIR doesn't exist.
if [ ! -d "$EXAMPLEDIR" ]; then
    # download examples
    wget https://www.dropbox.com/s/mvuhtgpnenmn5pg/examples.tar.gz?dl=1 -O examples.tar.gz
    tar -xzf examples.tar.gz
fi

# render the spheres with blender
echo "Rendering spheres..."
blender -b --python generate_spheres.py > /dev/null
echo "Saved rendered spheres."

# generate training dataset for classification
echo "Generating training dataset for classification..."
./generate_dataset_multithreaded.py "train_dataset_classification_500K_100batch" 500000 100 "0.5" 
echo "Generating training dataset for classification... Done."

# generate testing dataset for classification
echo "Generating testing dataset for classification..."
./generate_dataset_multithreaded.py "test_dataset_classification_200K_100batch" 200000 100 "0.5" 
echo "Generating testing dataset for classification... Done."

# generate training dataset for regression
echo "Generating training dataset for regression..."
./generate_dataset_multithreaded.py "train_dataset_regression_500k_100batch" 500000 100 "1.0" 
echo "Generating training dataset for regression... Done."

# generate testing dataset for regression
echo "Generating testing dataset for regression..."
./generate_dataset_multithreaded.py "test_dataset_regression_200k_100batch" 200000 100 "1.0" 
echo "Generating testing dataset for regression... Done."

