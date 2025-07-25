import torch

class ProteinVocabulary:
    def __init__(self):
        self.ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
        self.PAD_TOKEN = 20
        self.VOCAB_SIZE = 21  # 20 AA + PAD
        
        self.vocab_to_id = {aa: i for i, aa in enumerate(self.ALPHABET)}
        self.id_to_vocab = {i: aa for i, aa in enumerate(self.ALPHABET)}
        self.id_to_vocab[self.PAD_TOKEN] = '<PAD>'
    
    def encode_sequence(self, seq, max_length):
        """Encode an amino acid sequence into a tensor of IDs with padding."""
        ids = [self.vocab_to_id[aa] for aa in seq]
        
        if len(ids) > max_length:
            raise ValueError('sequence length (number of AA) is strictly greater than max length')
        else:
            pad_length = max_length - len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            padding = torch.full((pad_length,), self.PAD_TOKEN, dtype=torch.long)
            seq_padded = torch.cat([ids_tensor, padding])
            return seq_padded
    
    def decode_sequence(self, tokens):
        """Decode a tensor of IDs into an amino acid sequence."""
        result = []
        for tok in tokens:
            token_id = int(tok)
            if token_id == self.PAD_TOKEN:
                break  # Stop at first PAD
            else:
                result.append(self.id_to_vocab[token_id])
        return ''.join(result)
    
    def encode_batch(self, sequences, max_length):
        """Encode a batch of sequences with padding."""
        encoded_sequences = []
        for seq in sequences:
            encoded_seq = self.encode_sequence(seq, max_length)
            encoded_sequences.append(encoded_seq)
        return torch.stack(encoded_sequences)
    
    def decode_batch(self, encoded_sequences):
        """Decode a batch of encoded sequences."""
        decoded_sequences = []
        for seq in encoded_sequences:
            decoded_seq = self.decode_sequence(seq)
            decoded_sequences.append(decoded_seq)
        return decoded_sequences