mkdir ~/data
echo "Copying LJSpeech-1.1 dataset"
cp -r /blob/v-jcong/data/LJSpeech-1.1 ~/data
echo "Done"
mkdir -p ~/data/LJSpeech-1.1-fs2
echo "Copying LJSpeech-1.1 pitch dataset"
cp -r /blob/v-jcong/data/LJSpeech-1.1-fs2/f0 ~/data/LJSpeech-1.1-fs2
echo "Done"