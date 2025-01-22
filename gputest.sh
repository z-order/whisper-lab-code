#!/bin/bash

NUMOF_THREADS=6
LOG=true # or false
ENDPOINT="http://127.0.0.1:8000/api/v1/audio/transcriptions"
#ENDPOINT="http://ip-api.com"
#ENDPOINT="https://api.example.com/endpoint"

# Define the curl function
make_request_console() {
    local endpoint=$1
    local thread_num=$2
    echo "Thread $thread_num starting..."
    echo "Making request to $endpoint"
    # Make the curl request and print response
    response=$(curl -s $endpoint)
    echo "Thread $thread_num response: $response"
    echo "Thread $thread_num completed"
}

# Define the curl function
make_request_log() {
    cmdshell() {
        local thread_num=$1
        local f="output_${thread_num}.log"
        shift  # Remove first parameter (thread_num)
        local cmd=$@ # all parameters
        echo -n "$(date +%Y/%m/%d-%H:%M:%S) " >> "$f"
        eval "$cmd" >> "$f" 2>&1  # Redirect both stdout and stderr
    }    
    local endpoint=$1
    local thread_num=$2
    cmdshell $thread_num echo "Thread $thread_num starting..."
    cmdshell $thread_num echo "Making request to $endpoint"
    # Make the curl request and save response to file
    cmdshell $thread_num echo "Thread $thread_num response:"
    response=$(
        curl -s "$endpoint" \
            -H "accept: application/json" \
            -H "Content-Type: multipart/form-data" \
            -F file=@../audio/training-audio-$thread_num.mp3 \
            -F "timestamp_granularities[]=segment" \
            -F model="whisper-1" \
            -F response_format="verbose_json" \
            -X POST 
    )
    # response=$(
    #     curl -s "$endpoint" \
    #         -H "accept: application/json" \
    #         -X GET
    # )
    cmdshell $thread_num echo $response
    cmdshell $thread_num echo "Thread $thread_num completed"
}

# Export the function so it's available to xargs
export -f make_request_console
export -f make_request_log

echo "Press [CTRL+C] to stop.."

# Run NUMOF_THREADS parallel instances
if [ "$LOG" = "true" ]; then
    echo "Running with logging enabled"
    while :
    do
        seq $NUMOF_THREADS | xargs -P $NUMOF_THREADS -I {} bash -c "make_request_log $ENDPOINT {}"
        sleep 1; printf ...
    done
else
    echo "Running with logging disabled"
    while :
    do
        seq $NUMOF_THREADS | xargs -P $NUMOF_THREADS -I {} bash -c "make_request_console $ENDPOINT {}"
        sleep 1; printf ...
    done
fi
