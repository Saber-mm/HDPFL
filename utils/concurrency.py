import threading

def multithreads(threads):
    # create the threads
    for t in threads:
        t.start()
        
    # If any thread created, joins them.
    for t in threads:
        t.join()
        
