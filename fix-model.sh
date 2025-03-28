#!/bin/bash
# save this as fix-model.sh and run with: bash fix-model.sh

# Update the config file
sed -i 's/neulab\/stella-400m-v5/Snowflake\/snowflake-arctic-embed-xs/g' config/my_config.yaml

# Clean any cached data
rm -rf storage/vectors/*

# Re-initialize 
knowledge-commons init --config config/my_config.yaml --force

echo "Configuration fixed and system reinitialized"