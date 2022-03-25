import h5py as h5
import numpy as np
import cupy as cp
import tensorflow as tf
import timeit
import matplotlib.pyplot as plt

tf.config.experimental.enable_tensor_float_32_execution(False)

barWidth = 0.2

np_times = []
cp_times = []
tf_times = []
tfjit_times = []

with h5.File("bench.h5", "r") as bench:
    cx = bench["cx"][()]
    delta_cx = bench["delta_cx"][()]
    e_accept = bench["e_accept"][()]


def np_inplace_add():
    cx[e_accept] += delta_cx[e_accept]
    return cx.copy()


cx_cp = cp.array(cx)
dcx_cp = cp.array(delta_cx)
acc_cp = cp.array(e_accept)


def cp_inplace_add():
    cx_cp[acc_cp] += dcx_cp[acc_cp]
    return cx_cp.get()


sel_add = cp.ElementwiseKernel(
    "float32 delta, bool sel", "float32 dest", "if(sel) dest += delta", "sel_add"
)


def cp_inplace_add_custom():
    sel_add(dcx_cp, acc_cp[:, None, None], cx_cp)
    return cx_cp.get()


cx_tf = tf.constant(cx)
dcx_tf = tf.constant(delta_cx)
acc_tf = tf.constant(e_accept)


def tf_inplace_add_raw():
    global cx_tf
    i = tf.cast(tf.experimental.numpy.nonzero(acc_tf)[0], tf.int32)
    cx_tf = tf.raw_ops.InplaceAdd(x=cx_tf, i=i, v=tf.gather(dcx_tf, i))
    return cx_tf.numpy()


def tf_inplace_add_where():
    global cx_tf
    cx_tf = tf.where(acc_tf[:, None, None], cx_tf + dcx_tf, cx_tf)
    return cx_tf.numpy()


@tf.function(jit_compile=True)
def bool_inplace_add_3(x, b, y):
    return tf.where(b[:, None, None], x + y, x)


def tf_inplace_add_jit():
    global cx_tf
    cx_tf = bool_inplace_add_3(cx_tf, acc_tf, dcx_tf)
    return cx_tf.numpy()


N = 100

print("Benchmarking inplace add...")
dur_np = timeit.Timer(np_inplace_add).timeit(N)

dur_raw = timeit.Timer(tf_inplace_add_raw).timeit(N)
dur_where = timeit.Timer(tf_inplace_add_where).timeit(N)
dur_jit = timeit.Timer(tf_inplace_add_jit).timeit(N)

dur_cp = timeit.Timer(cp_inplace_add).timeit(N)
dur_cp_custom = timeit.Timer(cp_inplace_add_custom).timeit(N)

print("NumPy:", dur_np)
print("CuPy:", dur_cp)
print("CuPy kernel:", dur_cp_custom)
print("TF.InplaceAdd:", dur_raw)
print("TF.where:", dur_where)
print("TF.where+jit:", dur_jit)

np_times.append(dur_np)
cp_times.append(dur_cp_custom)
tf_times.append(dur_raw)
tfjit_times.append(dur_jit)

with h5.File("bench2.h5", "r") as bench:
    vA = bench["vA"][()]
    vc = bench["vc"][()]
    fmu = bench["fmu"][()]
    A = bench["A"][()]
    beta = bench["beta"][()]


def np_mul1():
    Δcx = -np.transpose(beta @ vc, (0, 2, 1)) @ (vc @ A)
    return Δcx.copy()


def np_mul2():
    Δz = -(fmu[:, None, :] @ vA).squeeze()
    return Δz.copy()


vA_cp = cp.array(vA)
vc_cp = cp.array(vc)
fmu_cp = cp.array(fmu)
A_cp = cp.array(A)
beta_cp = cp.array(beta)


def cp_mul1():
    Δcx = -cp.transpose(beta_cp @ vc_cp, (0, 2, 1)) @ (vc_cp @ A_cp)
    return Δcx.get()


def cp_mul2():
    Δz = -(fmu_cp[:, None, :] @ vA_cp).squeeze()
    return Δz.get()


vA_tf = tf.constant(vA)
vc_tf = tf.constant(vc)
fmu_tf = tf.constant(fmu)
A_tf = tf.constant(A)
beta_tf = tf.constant(beta)


def tf_mul1():
    Δcx = -tf.transpose(beta_tf @ vc_tf, (0, 2, 1)) @ (vc_tf @ A_tf)
    return Δcx.numpy()


def tf_mul2():
    Δz = -tf.squeeze(fmu_tf[:, None, :] @ vA_tf)
    return Δz.numpy()


@tf.function(jit_compile=True)
def mul1_jit(c_vec, A, beta):
    Δcx = -tf.transpose(beta @ c_vec, (0, 2, 1)) @ (c_vec @ A)
    return Δcx


@tf.function(jit_compile=True)
def mul2_jit(A_vec, fmu):
    Δz = -tf.squeeze(fmu[:, None, :] @ A_vec)
    return Δz


def tf_mul1_jit():
    Δcx = mul1_jit(vc_tf, A_tf, beta_tf)
    return Δcx.numpy()


def tf_mul2_jit():
    Δz = mul2_jit(vA_tf, fmu_tf)
    return Δz.numpy()


print("\nBenchmarking mul1...")
dur_np = timeit.Timer(np_mul1).timeit(N)
dur_raw = timeit.Timer(tf_mul1).timeit(N)
dur_jit = timeit.Timer(tf_mul1_jit).timeit(N)
dur_cp = timeit.Timer(cp_mul1).timeit(N)
print("NumPy:", dur_np)
print("CuPy:", dur_cp)
print("TF:", dur_raw)
print("TF.jit:", dur_jit)

np_times.append(dur_np)
cp_times.append(dur_cp)
tf_times.append(dur_raw)
tfjit_times.append(dur_jit)

print("\nBenchmarking mul2...")
dur_np = timeit.Timer(np_mul2).timeit(N)
dur_raw = timeit.Timer(tf_mul2).timeit(N)
dur_jit = timeit.Timer(tf_mul2_jit).timeit(N)
dur_cp = timeit.Timer(cp_mul2).timeit(N)
print("NumPy:", dur_np)
print("CuPy:", dur_cp)
print("TF:", dur_raw)
print("TF.jit:", dur_jit)

np_times.append(dur_np)
cp_times.append(dur_cp)
tf_times.append(dur_raw)
tfjit_times.append(dur_jit)

br_np = np.arange(3)
br_cp = br_np + barWidth
br_tf = br_cp + barWidth
br_jit = br_tf + barWidth

plt.bar(br_np, np_times, width=barWidth, label="NumPy")
plt.bar(br_cp, cp_times, width=barWidth, label="CuPy")
plt.bar(br_tf, tf_times, width=barWidth, label="Tensorflow")
plt.bar(br_jit, tfjit_times, width=barWidth, label="TF.jit")

plt.ylabel("Time/s")
plt.xticks(br_cp + barWidth / 2, ["inplace add", "mul1", "mul2"])

plt.legend()
plt.savefig("bench.pdf")
