# curblasContext

## Description

Internal cuRBLAS context structure

## Functions

### curblasContext

```c
 curblasContext()
```


## Variables

### stream

```c
cudaStream_t stream
```

### ownsStream

```c
bool ownsStream
```

### rng

```c
curblasRngState* rng
```

Custom random number generator state optimized for CUDA devices

### seed

```c
unsigned long long seed
```

### accuracy

```c
curblasAccuracy_t accuracy
```

### defaultSketchType

```c
curblasSketchType_t defaultSketchType
```

### mathMode

```c
curblasMath_t mathMode
```

### deviceId

```c
int deviceId
```

### version

```c
int version
```