task(){
    while [ $1 -ne 1 ]
	do
        rm -r "${3}run${2}"
        # echo "python run_single_ACTIN_SMALL_BOX_freezeActin.py -b ${8} -r $2 -t $5 -o $3 -m 0 -c $4 -p 3 -a ${9} -x $6" >> file.txt
		python run_single_GENES_STAGES.py -b ${8} -r ${2} -t ${5} -o ${3} -m 0 -c ${4} -p ${10} -a ${9} -x ${6};
		if test -f "${7}"; then
            rm ${7}
			break
		fi
	done
	
}

# initialize a semaphore with a given number of tokens
open_sem(){
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for((;i>0;i--)); do
        printf %s 000 >&3
    done
}

# run the given command asynchronously and pop/push tokens
run_with_lock(){
    local x
    # this read waits until there is something to read
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
     ( "$@"; )
    # push the return code of the command to the semaphore
    printf '%.3d' $? >&3
    )&
}

N=9
open_sem $N
PROMOTERS=(1 2 3)
# ACTIVATIONS=(1 2 3 4 5 6 7 8 9 10 15 20 25 30 100)
ACTIVATIONS=(1 5 10 15 20 25 30 50 75 100)
# ACTIVATIONS=(1 5 10 15 20 25 30 50 75 100)
THRESHOLDS=(1 10 20 30 40 50 60 70 75 80 90 100 370)

for PROMOTER in ${PROMOTERS[@]}; do
    for ACTIVATION in ${ACTIVATIONS[@]}; do
        for THRESHOLD in ${THRESHOLDS[@]}; do
            for r in $(seq $1 $2); do
                FILENAME="box$3/${4}_Promoter${PROMOTER}_Threshold${THRESHOLD}_Act${ACTIVATION}/run${r}/figures/ser5p_cluster.pdf"
                # FILENAME="box${3}/${4}_Promoter${PROMOTER}_Threshold${THRESHOLD}_Act${ACTIVATION}/parallel_counter/run${r}.txt"
                FILEEXISTS=0
                if test -f "$FILENAME"; then
                    FILEEXISTS=1
                    echo "$FILENAME exists already."
                fi
                run_with_lock task $FILEEXISTS $r "box${3}/${4}_Promoter${PROMOTER}_Threshold${THRESHOLD}_Act${ACTIVATION}/" $4 $2 ${THRESHOLD} ${FILENAME} $3 ${ACTIVATION} ${PROMOTER}
            done
        done
    done
done
