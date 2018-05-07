#!/bin/bash

files=(
#2011_10_03_calib.zip
00:2011_10_03_drive_0027
01:2011_10_03_drive_0042
02:2011_10_03_drive_0034
#2011_09_26_calib.zip
03:2011_09_26_drive_0067
#2011_09_30_calib.zip
04:2011_09_30_drive_0016
05:2011_09_30_drive_0018
06:2011_09_30_drive_0020
07:2011_09_30_drive_0027
08:2011_09_30_drive_0028
09:2011_09_30_drive_0033
10:2011_09_30_drive_0034
)

mkdir 'images'

for i in ${files[@]}; do
        if [ ${i:(-3)} != "zip" ]
        then
                rename=${i:0:2}
                shortname=${i:3}'_sync.zip'
                fullname=${i:3}'/'${i:3}'_sync.zip'

        else
                $i=${i:(-3)}
                shortname=$i
                fullname=$i
        fi
        echo "Downloading: "$shortname

        wget 'http://kitti.is.tue.mpg.de/kitti/raw_data/'$fullname
        unzip -o $shortname
        rm $shortname

        # Remove image00 image01 image02, rename dir
        if [ ${i:(-3)} != "zip" ]
        then
                dirn=${i:3:10}'/'${i:3}'_sync''/'
                rm -r $dirn'image_00/' $dirn'image_01/' $dirn'image_02/' $dirn'velodyne_points/'
                mv $dirn 'images/'$rename
        fi
done

