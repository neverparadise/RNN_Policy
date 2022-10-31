# RNN Policy
- encoder network shared? 
- GRU -> FC
- FC -> GRU -> FC
- FC -> GRU
- num_layers > 2

1. hidden_shape = (num_rnn_layers, batch_size, hidden_size)
2. intput_shape = (seq_len, batch_size, (input_size,))
3. eval input->  (1, 1, (input_size,))
4.  train input -> (seq_len, batch_size, input_size)
    we need just first hidden_state

# RolloutBuffer
- seq_len = 1 (one transition)
- seq_len = L (arbitrary length)
- seq_len = H (max episode length)

####(1) add sample

####(2) sample batch
- randomly sample transitions: 전체 버퍼에서 랜덤으로 트랜지션을 가져오면 됨
- sample whole episodes: 랜덤으로 에피소드들을 선택해서 전체 trajectory를 마스킹해서 가져오면 됨
- sample arbitrary seqeunce: 랜덤으로 에피소드들을 선택하고, 그 내부에서 시퀀스를 랜덤으로 샘플링해야함