mkdir -p ~/data/LJSpeech-1.1
echo "Copying LJSpeech-1.1 dataset"
cp -r /blob/v-jcong/data/LJSpeech-1.1/raw ~/data/LJSpeech-1.1
echo "Done"
mkdir -p ~/data/LJSpeech-1.1-fs2
echo "Copying LJSpeech-1.1 pitch dataset"
cp -r /blob/v-jcong/data/LJSpeech-1.1-fs2/f0 ~/data/LJSpeech-1.1-fs2
echo "Done"
cp -r ~/ms_container_init/dataset/filelists ~
echo "Success!"