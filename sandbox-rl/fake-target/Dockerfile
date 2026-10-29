FROM alpine:latest
RUN mkdir -p /data/important_folder && \
    echo "dummy file 1" > /data/important_folder/file1.txt && \
    echo "dummy file 2" > /data/important_folder/file2.txt
# simple long-running process so the container stays up
CMD ["tail", "-f", "/dev/null"]
