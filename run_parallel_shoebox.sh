task(){
    while [ $1 -ne 1 ]
	do
        if [ -d "${3}run${2}" ]; then
            rm -r "${3}run${2}"
        fi
		python run_single_shoebox.py -b ${8} -r ${2} -t ${5} -o ${3} -m 0 -c ${4} -p ${12} -a ${9} -x ${6} -z ${10} -n ${11};
		if test -f "${7}"; then
            rm ${7}
			break
		fi
	done
	
}

# Initialize a semaphore with a given number of tokens
open_sem(){
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for((;i>0;i--)); do
        printf %s 000 >&3
    done
}

# Run the given command asynchronously and pop/push tokens
run_with_lock(){
    local x
    # This read waits until there is something to read
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
     ( "$@"; )
    # Push the return code of the command to the semaphore
    printf '%.3d' $? >&3
    )&
}

# No. of parallel tasks you want to run
N=8 
open_sem $N

# Parameters from the bash script's arguments
REPEAT_START=${1}
REPEAT_END=${2}
BOX_LENGTH=${3}
CONDITION=${4}

# Parameters to loop over
THRESHOLD=80
ACTIVATION=30
PROMOTER=3
ACTIN=0
PNUC=0.01
decimals=${PNUC#*.}
PNUCX=${#decimals}

for r in $(seq ${REPEAT_START} ${REPEAT_END}); do
    if [ ${ACTIN} != 0 ]; then
        if [ ${CONDITION} != "LatB" ]; then
            FOLDER="${CONDITION}_Actin${ACTIN}_Threshold${THRESHOLD}_Act${ACTIVATION}_pnuc${PNUCX}d"
        else
            FOLDER="${CONDITION}_Actin${ACTIN}_Threshold${THRESHOLD}_Act${ACTIVATION}"
        fi
    else
        FOLDER="${CONDITION}_Promoter${PROMOTER}_Threshold${THRESHOLD}_Act${ACTIVATION}"
    fi
    # Make sure the following file is something you would have for a run that's definitely finished
    FILENAME="box${BOX_LENGTH}/${FOLDER}/run${r}/figures/ser5p_cluster.pdf"
    FILEEXISTS=0
    if test -f "$FILENAME"; then
        FILEEXISTS=1
        echo "$FILENAME exists already."
    fi
    run_with_lock task $FILEEXISTS $r "box${BOX_LENGTH}/${FOLDER}/" ${CONDITION} ${REPEAT_END} ${THRESHOLD} ${FILENAME} ${BOX_LENGTH} ${ACTIVATION} ${ACTIN} ${PNUC} ${PROMOTER}
done
