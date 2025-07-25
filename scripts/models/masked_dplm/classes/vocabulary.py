import torch

class ProteinVocabulary:
    def __init__(self):
        self.ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
        self.MASK_TOKEN = 20
        self.PAD_TOKEN = 21
        self.VOCAB_SIZE = 22  # 20 AA + 2 tokens spéciaux
        
        self.vocab_to_id = {aa: i for i, aa in enumerate(self.ALPHABET)}
        self.id_to_vocab = {i: aa for i, aa in enumerate(self.ALPHABET)}
        self.id_to_vocab[self.PAD_TOKEN] = '<PAD>'
        self.id_to_vocab[self.MASK_TOKEN] = 'X'
    
    def encode_sequence(self, seq, max_length):
        """Encode une séquence d'acides aminés en tensor d'IDs avec padding."""
        ids = [self.vocab_to_id[aa] for aa in seq]
        
        if len(ids) > max_length:
            raise ValueError('len seq (nombre AA) superieur strict a max length')
        else:
            pad_length = max_length - len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            padding = torch.full((pad_length,), self.PAD_TOKEN, dtype=torch.long)
            seq_padded = torch.cat([ids_tensor, padding])
            return seq_padded
    
    def decode_sequence(self, tokens):
        """Décode un tensor d'IDs en séquence d'acides aminés."""
        result = []
        for tok in tokens:
            token_id = int(tok)
            if token_id == self.PAD_TOKEN:
                break  # Arrêt au premier PAD
            elif token_id == self.MASK_TOKEN:
                result.append('X')
            else:
                result.append(self.id_to_vocab[token_id])
        return ''.join(result)
    
    def encode_batch(self, sequences, max_length):
        """Encode un batch de séquences avec padding."""
        encoded_sequences = []
        for seq in sequences:
            encoded_seq = self.encode_sequence(seq, max_length)
            encoded_sequences.append(encoded_seq)
        return torch.stack(encoded_sequences)
    
    def decode_batch(self, encoded_sequences):
        """Décode un batch de séquences encodées."""
        decoded_sequences = []
        for encoded_seq in encoded_sequences:
            decoded_seq = self.decode_sequence(encoded_seq)
            decoded_sequences.append(decoded_seq)
        return decoded_sequences