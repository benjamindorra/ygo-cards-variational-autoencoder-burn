#!/bin/bash

array=$(ls card_images_small/train | tail -10)
for elem in $array ; do mv card_images_small/train/$elem card_images_small/test ; done
