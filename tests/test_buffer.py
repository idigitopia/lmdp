import pytest
import numpy as np



def gen_random_transition(buffer):
    return [np.random.rand(*buffer.state_shape), 
           np.random.rand(*buffer.action_shape), 
           np.random.rand(*buffer.state_shape), 
           np.random.rand(1)[0],
           bool(np.random.randint(2))]


@pytest.fixture
def standard_buffer_instance():
    """
    get empty buffer of default state shape and action shape.
    """
    from lmdp.data.buffer import StandardBuffer
    import numpy as np 
    
    # instantiate StandardBuffer 
    state_shape = [2,2]
    action_shape = [4]
    buffer_size = 1000
    device =  "cpu"
    standard_buffer = StandardBuffer(state_shape = state_shape,
                          action_shape = action_shape,
                          buffer_size = buffer_size,
                          device = device)

    return standard_buffer


@pytest.mark.buffer
@pytest.mark.standard_buffer
def test_standard_buffer(standard_buffer_instance):
    buffer = standard_buffer_instance
        
    # fill the buffer
    for i in range(buffer.max_size):
        buffer.add(*gen_random_transition(buffer))
        assert i + 1 == len(buffer)
    
    # fill the buffer some more
    for i in range(int(buffer.max_size/2)):        
        buffer.add(*gen_random_transition(buffer))
        # make sure FIFO replacement of the states. 
        assert buffer.ptr == i + 1
        assert len(buffer) == buffer.max_size
        
    assert buffer.state_shape == [2,2]