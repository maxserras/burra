source ./venv/bin/activate
pip install -r requirements.txt

# Initialize LASER path to the root of the LASER repository (https://github.com/facebookresearch/LASER)
export LASER=/data/tools/laser3/
# If this is the first time you run LASER, you need to download the models and external tools
# $LASER/nllb/download_models.sh
# $LASER/install_extrenal_tools.sh  
# Might have to manually install FastBPE (check if it failed in the previous step). 
# 1. Go to $LASER/tools/fastBPE and run "g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast"
# 2. Run (using this venv) "python setup.py install"


# The source and target filenames must have the same basename, but different extensions.
# We have combined the task, input and output into a single line to make it easier to align.
bname="alpaca"
lsrc="es" # https://huggingface.co/datasets/somosnlp/somos-clean-alpaca-es
ltrg="en" # https://github.com/tloen/alpaca-lora/blob/main/alpaca_data_cleaned.json


# Create ids. It is a dummy step, but it is needed. For each line in the input file, it creates a line in the output file with language code-line number.
paste -d' ' <(seq 1 $(wc -l < $bname.$lsrc)) | sed 's/^/${lsrc}-/' > $bname.ids.$lsrc
paste -d' ' <(seq 1 $(wc -l < $bname.$ltrg)) | sed 's/^/${ltrg}-/' > $bname.ids.$ltrg

# encoder and bpe codes 
model_dir="${LASER}/models"
# You might download the models first:
# wget https://dl.fbaipublicfiles.com/laser/models/bilstm.93langs.2018-12-26.pt -P $model_dir
# wget https://dl.fbaipublicfiles.com/laser/models/93langs.fcodes -P $model_dir
# wget https://dl.fbaipublicfiles.com/laser/models/93langs.fvocab -P $model_dir
encoder="${model_dir}/bilstm.93langs.2018-12-26.pt"
bpe_codes="${model_dir}/93langs.fcodes"

# This is a slight modification of the LASER to simplify the process of embedding and mining bitexts.
Embed() {
    ll=$2
    txt="$1.$ll"
    enc="$1.enc.$ll"
    echo "Embedding ${txt} to ${enc}"
    echo "Encoder: ${encoder}"
    if [ ! -s ${enc} ] ; then
        cat ${txt} | python3 ${LASER}/source/embed.py \
        --encoder ${encoder} \
        --token-lang ${ll} \
        --bpe-codes ${bpe_codes} \
        --output ${enc} \
        --verbose 
    fi
}

Mine () {
  basename=$1
  l1=$2
  l2=$3
  cand="${basename}.candidates.tsv"
  if [ ! -s ${cand} ] ; then
    python3 ${LASER}/source/mine_bitexts.py \
       ${basename}.${l1} ${basename}.${l2} \
       --src-lang ${l1} --trg-lang ${l2} \
       --src-embeddings ${basename}.enc.${l1} --trg-embeddings ${basename}.enc.${l2} \
       --unify --mode mine --retrieval max --margin ratio -k 4  \
       --output ${cand} \
       --verbose --gpu
  fi
}

echo -e "\Mining bitexts for ${lsrc}-${ltrg} for ${source_file} and ${target_file}"
Embed ${bname} ${lsrc}
Embed ${bname} ${ltrg}

Mine ${bname} ${lsrc} ${ltrg}

if [ ! -s ${bname}.log ] ; then
python3 $LASER/tasks/bucc/bucc.py \
        --src-lang ${lsrc} --trg-lang ${ltrg} \
        --bucc-texts ${bname} \
        --bucc-ids ${bname}.ids \
        --candidates ${bname}.candidates.tsv \
        --output ${bname}.bucc \
        --verbose --threshold 0.01 | tee ${bname}.log
fi

