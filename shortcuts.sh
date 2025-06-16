export FYP='/vol/bitbucket/bl1821'
export DORADO="$FYP/dorado-1.0.0-linux-x64/bin"
export SQUIGULATOR="$FYP/squigulator"
export BONITO="$FYP//bonito_env/lib/python3.10/site-packages/bonito/models"

MODEL_PRE="$DORADO/dna_r10.4.1_e8.2_400bps_"
export FAST="${MODEL_PRE}fast@v5.0.0"
export HAC="${MODEL_PRE}hac@v5.0.0"
export SUP="${MODEL_PRE}sup@v5.0.0"

BONITO_MODEL_PRE="$BONITO/dna_r10.4.1_e8.2_400bps_"
export BFAST="${BONITO_MODEL_PRE}fast@v5.0.0"
export BHAC="${BONITO_MODEL_PRE}hac@v5.0.0"
export BSUP="${BONITO_MODEL_PRE}sup@v5.0.0"

alias squigulator='$SQUIGULATOR/squigulator'
alias dorado='$DORADO/dorado'
alias fyp='cd $FYP && source pyenv/bin/activate'
alias bnt='cd $FYP && source bonito_env/bin/activate'
alias aws='~/aws-cli/v2/2.27.24/bin/aws'
awsont() {
  local dir="${1:-}"
  local cmd="aws s3 ls --no-sign-request s3://ont-open-data/$dir"
  echo "Running the command: $cmd"
  eval "$cmd"
}