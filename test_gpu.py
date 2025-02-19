import tensorflow as tf


def show_devices():
    # List physical devices
    physical_devices = tf.config.list_physical_devices()
    print("Physical devices:", physical_devices)

    # List logical devices
    logical_devices = tf.config.list_logical_devices()
    print("Logical devices:", logical_devices)


def test_tensors():
    import tensorflow as tf
    import time

    # Create two large random tensors
    a = tf.random.normal([10000, 10000])
    b = tf.random.normal([10000, 10000])

    avg_time = 0
    for _ in range(100):
        # Perform matrix multiplication on GPU
        start_time = time.time()
        c = tf.matmul(a, b)
        end_time = time.time()

        avg_time += end_time - start_time
    print(f"Matrix multiplication on GPU took {avg_time/1000:.2f} seconds")


if __name__ == "__main__":
    test_tensors()
