#include <vector>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <iostream> 
#include <fstream>
#include <cstdint>

using namespace std;

inline uint64_t pack_pair(int a, int b) {
    return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
}

struct SplitMix64Hash
{
	size_t operator()(uint64_t cle) const
	{
		uint64_t y = cle + 0x9e3779b97f4a7c15 ;
		y = ( y ^ ( y >> 30 ) ) * 0xbf58476d1ce4e5b9;
		y = ( y ^ ( y >> 27 ) ) * 0x94d049bb133111eb;
		return  static_cast<size_t>(y ^ ( y >> 31 ));
	}
};

struct TokenNode {
	int token_id;
    int prev_idx;
    int next_idx;	int next_occ_idx = -1;
};

struct HeapNode {
	int freq;
	uint64_t id;
	bool operator<(const HeapNode& autre) const 
	{
		return freq < autre.freq;
	}
};
void merge(uint64_t target_id, int new_token_id, 
           vector<TokenNode>& articles, 
           unordered_map<uint64_t, vector<int>, SplitMix64Hash>& positions, 
           unordered_map<uint64_t, int, SplitMix64Hash>& freq, 
           priority_queue<HeapNode>& tas_max) 
{
    vector<uint64_t> modifications;
    
    int left_target = target_id >> 32;
    int right_target = target_id & 0xFFFFFFFF;

    for (int idx : positions[target_id]) {
        
        if (articles[idx].token_id != left_target) continue;
        
        int next_idx = articles[idx].next_idx;
        if (next_idx == -1 || articles[next_idx].token_id != right_target) continue;

        int prev_idx = articles[idx].prev_idx;
        int next_next_idx = articles[next_idx].next_idx;

        if (prev_idx != -1) {
            uint64_t old_prev_pair = pack_pair(articles[prev_idx].token_id, left_target);
            freq[old_prev_pair]--;
        }
        if (next_next_idx != -1) {
            uint64_t old_next_pair = pack_pair(right_target, articles[next_next_idx].token_id);
            freq[old_next_pair]--;
        }

        articles[idx].token_id = new_token_id;
        articles[idx].next_idx = next_next_idx;
        
        if (next_next_idx != -1) {
            articles[next_next_idx].prev_idx = idx;
        }

        articles[next_idx].token_id = -1;
        articles[next_idx].prev_idx = -1;
        articles[next_idx].next_idx = -1;

        if (prev_idx != -1) {
            uint64_t new_prev_pair = pack_pair(articles[prev_idx].token_id, new_token_id);
            freq[new_prev_pair]++;
            positions[new_prev_pair].push_back(prev_idx);
            modifications.push_back(new_prev_pair);
        }
        if (next_next_idx != -1) {
            uint64_t new_next_pair = pack_pair(new_token_id, articles[next_next_idx].token_id);
            freq[new_next_pair]++;
            positions[new_next_pair].push_back(idx);
            modifications.push_back(new_next_pair);
        }
    }

    sort(modifications.begin(), modifications.end());
    modifications.erase(unique(modifications.begin(), modifications.end()), modifications.end());

    for (uint64_t x : modifications) {
        tas_max.push({freq[x], x});
    }

    positions.erase(target_id); 
}


int main()
{
	ios::sync_with_stdio(0);
    cin.tie(0);

	vector<TokenNode> articles;
	unordered_map<uint64_t,int,SplitMix64Hash> freq;
	unordered_map<uint64_t,vector<int>,SplitMix64Hash> position;

	freq.reserve(5000000); 
	position.reserve(5000000);

	int previous = -1;
	int current ;
	uint32_t longueur = 0;

	const int MAX_TOKENS = 200000000;

	articles.reserve(MAX_TOKENS);

	ifstream preprocessed_data("fineweb.txt");

	if (preprocessed_data)
	{
		while (preprocessed_data >> current)
	{

		if (longueur >= MAX_TOKENS) {
                break; 
            }

		if (current == -1)
		{
			if (previous == -1) continue;
			articles[longueur-1].next_idx = -1;
		}
		else
		{
			TokenNode node;
			node.next_idx = -1 ;
			node.token_id = current;

			if (previous != -1)
			{
				node.prev_idx = longueur - 1;
				articles[longueur - 1].next_idx = longueur;

				uint64_t id = pack_pair(previous, current);
				freq[id]++;
				position[id].push_back(longueur - 1);
				
			}

			else node.prev_idx = -1;

			articles.push_back(node);
			longueur++;
		}

		previous = current;
	}

	preprocessed_data.close();

	}

	else cerr << "Erreur à l'ouverture de l'entrée !" << endl;

	vector<HeapNode> constructeur; 
	constructeur.reserve(freq.size());

	for (const auto& [id, frequence] : freq) 
	{
		constructeur.push_back({frequence,id});
	}

	priority_queue<HeapNode> tas_max(constructeur.begin(), constructeur.end());

	uint64_t inverse_vocab[32000];

	uint16_t taille = 256;
	while (taille < 32000)
	{
		HeapNode node;
		uint64_t id;
		bool valid_found = false;

		while (!tas_max.empty()) {
			node = tas_max.top();
			tas_max.pop();
			id = node.id;
			
			if (node.freq == freq[id]) {
				valid_found = true;
				break;
			}	
		}

		if (!valid_found || node.freq < 2) break; 
	
		merge(id,taille,articles,position,freq, tas_max);
		inverse_vocab[taille] = id;
		taille++;
	}


	// Fichier utilisé pour l'encodeur et le décodeur en python
	
    ofstream out_merges("merges.txt");
    if (out_merges) {
        for (int i = 256; i < taille; i++) {
            uint64_t paire = inverse_vocab[i];
            int left_id = paire >> 32;
            int right_id = paire & 0xFFFFFFFF;
            out_merges << left_id << " " << right_id << " " << i << "\n";
        }
        out_merges.close();
    }

    unordered_map<int, vector<uint8_t>> vocab_bytes;
    
    for (int i = 0; i < 256; i++) {
        vocab_bytes[i] = { static_cast<uint8_t>(i) };
    }

    ofstream out_vocab("vocab.txt");
    if (out_vocab) {
        for (int i = 256; i < taille; i++) {
            uint64_t paire = inverse_vocab[i];
            int left_id = paire >> 32;
            int right_id = paire & 0xFFFFFFFF;
            
            vector<uint8_t> seq = vocab_bytes[left_id];
            vector<uint8_t> right_seq = vocab_bytes[right_id];
            seq.insert(seq.end(), right_seq.begin(), right_seq.end());
            
            vocab_bytes[i] = seq;
        }

        for (int i = 0; i < taille; i++) {
            out_vocab << i << ":";
            for (size_t j = 0; j < vocab_bytes[i].size(); j++) {
                out_vocab << static_cast<int>(vocab_bytes[i][j]);
                if (j < vocab_bytes[i].size() - 1) out_vocab << ",";
            }
            out_vocab << "\n";
        }
        out_vocab.close();
    }
 
    return 0;
}