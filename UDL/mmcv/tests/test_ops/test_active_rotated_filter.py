import numpy as np
import pytest
import torch

from mmcv.ops import active_rotated_filter

np_feature = np.array([[[[[-1.4934e-01, 1.1341e+00, -1.6241e-01],
                          [-1.0986e+00, -1.1463e+00, -1.3176e+00],
                          [1.4808e+00, 7.6572e-01, -1.4548e+00]]]],
                       [[[[1.9370e+00, 6.2799e-01, 2.5834e-02],
                          [-1.4242e+00, 7.6566e-01, 1.0015e+00],
                          [9.8669e-01, 4.1356e-01, 6.1068e-01]]]],
                       [[[[1.4565e+00, 1.4960e+00, 2.4339e-01],
                          [-2.2484e-01, 7.5942e-01, -8.1184e-01],
                          [-1.7077e+00, 1.0658e+00, 3.8311e-01]]]],
                       [[[[8.4734e-01, 1.0904e+00, 2.4356e+00],
                          [9.5822e-01, 2.2260e-01, -2.4450e-01],
                          [-1.5078e+00, 7.0902e-02, -1.5921e+00]]]],
                       [[[[2.1173e+00, -7.3524e-01, 1.8888e+00],
                          [1.0169e+00, 4.7033e-01, -1.0875e+00],
                          [-1.0736e+00, -5.2245e-01, -2.8733e-01]]]],
                       [[[[-5.6433e-01, 1.5835e+00, -1.5826e+00],
                          [-8.8974e-01, -4.3128e-01, -2.2423e-01],
                          [1.6552e-03, -1.7292e+00, 2.6639e-01]]]],
                       [[[[-1.2951e-01, 1.3493e+00, -1.9329e+00],
                          [5.6248e-01, -5.1189e-01, 1.3614e+00],
                          [3.3680e-01, -8.7148e-01, 5.0592e-01]]]],
                       [[[[1.6781e-02, -8.3929e-01, 1.2060e+00],
                          [-1.0764e+00, 4.7821e-01, 1.5342e+00],
                          [-4.4542e-01, -1.8606e+00, 3.0827e-01]]]]])

np_indices = np.array([[[[1, 2, 3, 6, 9, 8, 7, 4], [2, 3, 6, 9, 8, 7, 4, 1],
                         [3, 6, 9, 8, 7, 4, 1, 2]],
                        [[4, 1, 2, 3, 6, 9, 8, 7], [5, 5, 5, 5, 5, 5, 5, 5],
                         [6, 9, 8, 7, 4, 1, 2, 3]],
                        [[7, 4, 1, 2, 3, 6, 9, 8], [8, 7, 4, 1, 2, 3, 6, 9],
                         [9, 8, 7, 4, 1, 2, 3, 6]]]])

expected_output = np.array([[[[-1.4934e-01, 1.1341e+00, -1.6241e-01],
                              [-1.0986e+00, -1.1463e+00, -1.3176e+00],
                              [1.4808e+00, 7.6572e-01, -1.4548e+00]]],
                            [[[-1.0986e+00, -1.4934e-01, 1.1341e+00],
                              [1.4808e+00, -1.1463e+00, -1.6241e-01],
                              [7.6572e-01, -1.4548e+00, -1.3176e+00]]],
                            [[[1.4808e+00, -1.0986e+00, -1.4934e-01],
                              [7.6572e-01, -1.1463e+00, 1.1341e+00],
                              [-1.4548e+00, -1.3176e+00, -1.6241e-01]]],
                            [[[7.6572e-01, 1.4808e+00, -1.0986e+00],
                              [-1.4548e+00, -1.1463e+00, -1.4934e-01],
                              [-1.3176e+00, -1.6241e-01, 1.1341e+00]]],
                            [[[-1.4548e+00, 7.6572e-01, 1.4808e+00],
                              [-1.3176e+00, -1.1463e+00, -1.0986e+00],
                              [-1.6241e-01, 1.1341e+00, -1.4934e-01]]],
                            [[[-1.3176e+00, -1.4548e+00, 7.6572e-01],
                              [-1.6241e-01, -1.1463e+00, 1.4808e+00],
                              [1.1341e+00, -1.4934e-01, -1.0986e+00]]],
                            [[[-1.6241e-01, -1.3176e+00, -1.4548e+00],
                              [1.1341e+00, -1.1463e+00, 7.6572e-01],
                              [-1.4934e-01, -1.0986e+00, 1.4808e+00]]],
                            [[[1.1341e+00, -1.6241e-01, -1.3176e+00],
                              [-1.4934e-01, -1.1463e+00, -1.4548e+00],
                              [-1.0986e+00, 1.4808e+00, 7.6572e-01]]],
                            [[[1.9370e+00, 6.2799e-01, 2.5834e-02],
                              [-1.4242e+00, 7.6566e-01, 1.0015e+00],
                              [9.8669e-01, 4.1356e-01, 6.1068e-01]]],
                            [[[-1.4242e+00, 1.9370e+00, 6.2799e-01],
                              [9.8669e-01, 7.6566e-01, 2.5834e-02],
                              [4.1356e-01, 6.1068e-01, 1.0015e+00]]],
                            [[[9.8669e-01, -1.4242e+00, 1.9370e+00],
                              [4.1356e-01, 7.6566e-01, 6.2799e-01],
                              [6.1068e-01, 1.0015e+00, 2.5834e-02]]],
                            [[[4.1356e-01, 9.8669e-01, -1.4242e+00],
                              [6.1068e-01, 7.6566e-01, 1.9370e+00],
                              [1.0015e+00, 2.5834e-02, 6.2799e-01]]],
                            [[[6.1068e-01, 4.1356e-01, 9.8669e-01],
                              [1.0015e+00, 7.6566e-01, -1.4242e+00],
                              [2.5834e-02, 6.2799e-01, 1.9370e+00]]],
                            [[[1.0015e+00, 6.1068e-01, 4.1356e-01],
                              [2.5834e-02, 7.6566e-01, 9.8669e-01],
                              [6.2799e-01, 1.9370e+00, -1.4242e+00]]],
                            [[[2.5834e-02, 1.0015e+00, 6.1068e-01],
                              [6.2799e-01, 7.6566e-01, 4.1356e-01],
                              [1.9370e+00, -1.4242e+00, 9.8669e-01]]],
                            [[[6.2799e-01, 2.5834e-02, 1.0015e+00],
                              [1.9370e+00, 7.6566e-01, 6.1068e-01],
                              [-1.4242e+00, 9.8669e-01, 4.1356e-01]]],
                            [[[1.4565e+00, 1.4960e+00, 2.4339e-01],
                              [-2.2484e-01, 7.5942e-01, -8.1184e-01],
                              [-1.7077e+00, 1.0658e+00, 3.8311e-01]]],
                            [[[-2.2484e-01, 1.4565e+00, 1.4960e+00],
                              [-1.7077e+00, 7.5942e-01, 2.4339e-01],
                              [1.0658e+00, 3.8311e-01, -8.1184e-01]]],
                            [[[-1.7077e+00, -2.2484e-01, 1.4565e+00],
                              [1.0658e+00, 7.5942e-01, 1.4960e+00],
                              [3.8311e-01, -8.1184e-01, 2.4339e-01]]],
                            [[[1.0658e+00, -1.7077e+00, -2.2484e-01],
                              [3.8311e-01, 7.5942e-01, 1.4565e+00],
                              [-8.1184e-01, 2.4339e-01, 1.4960e+00]]],
                            [[[3.8311e-01, 1.0658e+00, -1.7077e+00],
                              [-8.1184e-01, 7.5942e-01, -2.2484e-01],
                              [2.4339e-01, 1.4960e+00, 1.4565e+00]]],
                            [[[-8.1184e-01, 3.8311e-01, 1.0658e+00],
                              [2.4339e-01, 7.5942e-01, -1.7077e+00],
                              [1.4960e+00, 1.4565e+00, -2.2484e-01]]],
                            [[[2.4339e-01, -8.1184e-01, 3.8311e-01],
                              [1.4960e+00, 7.5942e-01, 1.0658e+00],
                              [1.4565e+00, -2.2484e-01, -1.7077e+00]]],
                            [[[1.4960e+00, 2.4339e-01, -8.1184e-01],
                              [1.4565e+00, 7.5942e-01, 3.8311e-01],
                              [-2.2484e-01, -1.7077e+00, 1.0658e+00]]],
                            [[[8.4734e-01, 1.0904e+00, 2.4356e+00],
                              [9.5822e-01, 2.2260e-01, -2.4450e-01],
                              [-1.5078e+00, 7.0902e-02, -1.5921e+00]]],
                            [[[9.5822e-01, 8.4734e-01, 1.0904e+00],
                              [-1.5078e+00, 2.2260e-01, 2.4356e+00],
                              [7.0902e-02, -1.5921e+00, -2.4450e-01]]],
                            [[[-1.5078e+00, 9.5822e-01, 8.4734e-01],
                              [7.0902e-02, 2.2260e-01, 1.0904e+00],
                              [-1.5921e+00, -2.4450e-01, 2.4356e+00]]],
                            [[[7.0902e-02, -1.5078e+00, 9.5822e-01],
                              [-1.5921e+00, 2.2260e-01, 8.4734e-01],
                              [-2.4450e-01, 2.4356e+00, 1.0904e+00]]],
                            [[[-1.5921e+00, 7.0902e-02, -1.5078e+00],
                              [-2.4450e-01, 2.2260e-01, 9.5822e-01],
                              [2.4356e+00, 1.0904e+00, 8.4734e-01]]],
                            [[[-2.4450e-01, -1.5921e+00, 7.0902e-02],
                              [2.4356e+00, 2.2260e-01, -1.5078e+00],
                              [1.0904e+00, 8.4734e-01, 9.5822e-01]]],
                            [[[2.4356e+00, -2.4450e-01, -1.5921e+00],
                              [1.0904e+00, 2.2260e-01, 7.0902e-02],
                              [8.4734e-01, 9.5822e-01, -1.5078e+00]]],
                            [[[1.0904e+00, 2.4356e+00, -2.4450e-01],
                              [8.4734e-01, 2.2260e-01, -1.5921e+00],
                              [9.5822e-01, -1.5078e+00, 7.0902e-02]]],
                            [[[2.1173e+00, -7.3524e-01, 1.8888e+00],
                              [1.0169e+00, 4.7033e-01, -1.0875e+00],
                              [-1.0736e+00, -5.2245e-01, -2.8733e-01]]],
                            [[[1.0169e+00, 2.1173e+00, -7.3524e-01],
                              [-1.0736e+00, 4.7033e-01, 1.8888e+00],
                              [-5.2245e-01, -2.8733e-01, -1.0875e+00]]],
                            [[[-1.0736e+00, 1.0169e+00, 2.1173e+00],
                              [-5.2245e-01, 4.7033e-01, -7.3524e-01],
                              [-2.8733e-01, -1.0875e+00, 1.8888e+00]]],
                            [[[-5.2245e-01, -1.0736e+00, 1.0169e+00],
                              [-2.8733e-01, 4.7033e-01, 2.1173e+00],
                              [-1.0875e+00, 1.8888e+00, -7.3524e-01]]],
                            [[[-2.8733e-01, -5.2245e-01, -1.0736e+00],
                              [-1.0875e+00, 4.7033e-01, 1.0169e+00],
                              [1.8888e+00, -7.3524e-01, 2.1173e+00]]],
                            [[[-1.0875e+00, -2.8733e-01, -5.2245e-01],
                              [1.8888e+00, 4.7033e-01, -1.0736e+00],
                              [-7.3524e-01, 2.1173e+00, 1.0169e+00]]],
                            [[[1.8888e+00, -1.0875e+00, -2.8733e-01],
                              [-7.3524e-01, 4.7033e-01, -5.2245e-01],
                              [2.1173e+00, 1.0169e+00, -1.0736e+00]]],
                            [[[-7.3524e-01, 1.8888e+00, -1.0875e+00],
                              [2.1173e+00, 4.7033e-01, -2.8733e-01],
                              [1.0169e+00, -1.0736e+00, -5.2245e-01]]],
                            [[[-5.6433e-01, 1.5835e+00, -1.5826e+00],
                              [-8.8974e-01, -4.3128e-01, -2.2423e-01],
                              [1.6552e-03, -1.7292e+00, 2.6639e-01]]],
                            [[[-8.8974e-01, -5.6433e-01, 1.5835e+00],
                              [1.6552e-03, -4.3128e-01, -1.5826e+00],
                              [-1.7292e+00, 2.6639e-01, -2.2423e-01]]],
                            [[[1.6552e-03, -8.8974e-01, -5.6433e-01],
                              [-1.7292e+00, -4.3128e-01, 1.5835e+00],
                              [2.6639e-01, -2.2423e-01, -1.5826e+00]]],
                            [[[-1.7292e+00, 1.6552e-03, -8.8974e-01],
                              [2.6639e-01, -4.3128e-01, -5.6433e-01],
                              [-2.2423e-01, -1.5826e+00, 1.5835e+00]]],
                            [[[2.6639e-01, -1.7292e+00, 1.6552e-03],
                              [-2.2423e-01, -4.3128e-01, -8.8974e-01],
                              [-1.5826e+00, 1.5835e+00, -5.6433e-01]]],
                            [[[-2.2423e-01, 2.6639e-01, -1.7292e+00],
                              [-1.5826e+00, -4.3128e-01, 1.6552e-03],
                              [1.5835e+00, -5.6433e-01, -8.8974e-01]]],
                            [[[-1.5826e+00, -2.2423e-01, 2.6639e-01],
                              [1.5835e+00, -4.3128e-01, -1.7292e+00],
                              [-5.6433e-01, -8.8974e-01, 1.6552e-03]]],
                            [[[1.5835e+00, -1.5826e+00, -2.2423e-01],
                              [-5.6433e-01, -4.3128e-01, 2.6639e-01],
                              [-8.8974e-01, 1.6552e-03, -1.7292e+00]]],
                            [[[-1.2951e-01, 1.3493e+00, -1.9329e+00],
                              [5.6248e-01, -5.1189e-01, 1.3614e+00],
                              [3.3680e-01, -8.7148e-01, 5.0592e-01]]],
                            [[[5.6248e-01, -1.2951e-01, 1.3493e+00],
                              [3.3680e-01, -5.1189e-01, -1.9329e+00],
                              [-8.7148e-01, 5.0592e-01, 1.3614e+00]]],
                            [[[3.3680e-01, 5.6248e-01, -1.2951e-01],
                              [-8.7148e-01, -5.1189e-01, 1.3493e+00],
                              [5.0592e-01, 1.3614e+00, -1.9329e+00]]],
                            [[[-8.7148e-01, 3.3680e-01, 5.6248e-01],
                              [5.0592e-01, -5.1189e-01, -1.2951e-01],
                              [1.3614e+00, -1.9329e+00, 1.3493e+00]]],
                            [[[5.0592e-01, -8.7148e-01, 3.3680e-01],
                              [1.3614e+00, -5.1189e-01, 5.6248e-01],
                              [-1.9329e+00, 1.3493e+00, -1.2951e-01]]],
                            [[[1.3614e+00, 5.0592e-01, -8.7148e-01],
                              [-1.9329e+00, -5.1189e-01, 3.3680e-01],
                              [1.3493e+00, -1.2951e-01, 5.6248e-01]]],
                            [[[-1.9329e+00, 1.3614e+00, 5.0592e-01],
                              [1.3493e+00, -5.1189e-01, -8.7148e-01],
                              [-1.2951e-01, 5.6248e-01, 3.3680e-01]]],
                            [[[1.3493e+00, -1.9329e+00, 1.3614e+00],
                              [-1.2951e-01, -5.1189e-01, 5.0592e-01],
                              [5.6248e-01, 3.3680e-01, -8.7148e-01]]],
                            [[[1.6781e-02, -8.3929e-01, 1.2060e+00],
                              [-1.0764e+00, 4.7821e-01, 1.5342e+00],
                              [-4.4542e-01, -1.8606e+00, 3.0827e-01]]],
                            [[[-1.0764e+00, 1.6781e-02, -8.3929e-01],
                              [-4.4542e-01, 4.7821e-01, 1.2060e+00],
                              [-1.8606e+00, 3.0827e-01, 1.5342e+00]]],
                            [[[-4.4542e-01, -1.0764e+00, 1.6781e-02],
                              [-1.8606e+00, 4.7821e-01, -8.3929e-01],
                              [3.0827e-01, 1.5342e+00, 1.2060e+00]]],
                            [[[-1.8606e+00, -4.4542e-01, -1.0764e+00],
                              [3.0827e-01, 4.7821e-01, 1.6781e-02],
                              [1.5342e+00, 1.2060e+00, -8.3929e-01]]],
                            [[[3.0827e-01, -1.8606e+00, -4.4542e-01],
                              [1.5342e+00, 4.7821e-01, -1.0764e+00],
                              [1.2060e+00, -8.3929e-01, 1.6781e-02]]],
                            [[[1.5342e+00, 3.0827e-01, -1.8606e+00],
                              [1.2060e+00, 4.7821e-01, -4.4542e-01],
                              [-8.3929e-01, 1.6781e-02, -1.0764e+00]]],
                            [[[1.2060e+00, 1.5342e+00, 3.0827e-01],
                              [-8.3929e-01, 4.7821e-01, -1.8606e+00],
                              [1.6781e-02, -1.0764e+00, -4.4542e-01]]],
                            [[[-8.3929e-01, 1.2060e+00, 1.5342e+00],
                              [1.6781e-02, 4.7821e-01, 3.0827e-01],
                              [-1.0764e+00, -4.4542e-01, -1.8606e+00]]]])

expected_grad = np.array([[[[[8., 8., 8.], [8., 8., 8.], [8., 8., 8.]]]],
                          [[[[8., 8., 8.], [8., 8., 8.], [8., 8., 8.]]]],
                          [[[[8., 8., 8.], [8., 8., 8.], [8., 8., 8.]]]],
                          [[[[8., 8., 8.], [8., 8., 8.], [8., 8., 8.]]]],
                          [[[[8., 8., 8.], [8., 8., 8.], [8., 8., 8.]]]],
                          [[[[8., 8., 8.], [8., 8., 8.], [8., 8., 8.]]]],
                          [[[[8., 8., 8.], [8., 8., 8.], [8., 8., 8.]]]],
                          [[[[8., 8., 8.], [8., 8., 8.], [8., 8., 8.]]]]])


@pytest.mark.parametrize('device', [
    'cpu',
    pytest.param(
        'cuda',
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason='requires CUDA support')),
])
def test_active_rotated_filter(device):
    feature = torch.tensor(
        np_feature, dtype=torch.float, device=device, requires_grad=True)
    indices = torch.tensor(np_indices, dtype=torch.int, device=device)
    output = active_rotated_filter(feature, indices)
    output.backward(torch.ones_like(output))
    assert np.allclose(output.data.cpu().numpy(), expected_output, atol=1e-3)
    assert np.allclose(
        feature.grad.data.cpu().numpy(), expected_grad, atol=1e-3)
