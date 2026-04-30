#include <vector>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <iostream> 
#include <fstream>
#include <cstdint>

using namespace std;

inline uint64_t pack_pair(uint16_t a, int16_t b) {
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
    uint16_t token_id;
    int prev_idx;
    int next_idx;
	int next_occ_idx;
};

struct HeapNode {
	int freq;
	uint64_t id;
	bool operator<(const HeapNode& autre) const 
	{
		return freq < autre.freq;
	}
};

void merge(uint64_t& id,uint16_t& taille ,vector<TokenNode>& articles,unordered_map<uint64_t,vector<int>,SplitMix64Hash>& position,unordered_map<uint64_t,int,SplitMix64Hash>& freq, priority_queue<HeapNode>& tas_max )
{
	vector<uint64_t> modification;

	for (int x : position[id])
	{
		if (articles[x].token_id==-1) continue;

		int next = articles[x].next_idx;
		if (next == -1) continue; 
		if (pack_pair(articles[x].token_id, articles[next].token_id) != id) continue;
		

		int prev = articles[x].prev_idx;
		if (prev != -1)
		{
			uint64_t a = pack_pair(articles[prev].token_id,articles[x].token_id);
			freq[a]--;
			modification.push_back(a);

			uint64_t b = pack_pair(articles[prev].token_id,taille);
			freq[b]++;
			modification.push_back(b);
			position[b].push_back(prev);

		}
	
		if (articles[next].next_idx != -1)
		{
			uint64_t a = pack_pair(articles[next].token_id,articles[articles[next].next_idx].token_id);
			freq[a]--;
			modification.push_back(a);
			uint64_t b = pack_pair(taille,articles[articles[next].next_idx].token_id);
			freq[b]++;
			modification.push_back(b);
			position[b].push_back(x);

			articles[articles[next].next_idx].prev_idx = x;
				
		}
		articles[x].next_idx = articles[next].next_idx;
		articles[next].token_id = -1;
		

		articles[x].token_id = taille;

	}

	sort(modification.begin(),modification.end());
	modification.erase( unique(modification.begin(), modification.end()), modification.end() );

	for (uint64_t x : modification)
	{
		HeapNode node;
		node.freq = freq[x];
		node.id = x;
		tas_max.push(node);
	}
};


int main()
{
	ios::sync_with_stdio(0);
    cin.tie(0);

	vector<TokenNode> articles;
	unordered_map<uint64_t,int,SplitMix64Hash> freq;
	unordered_map<uint64_t,vector<int>,SplitMix64Hash> position;

	int previous = -1;
	int current ;
	uint32_t longueur = 0;

	const int MAX_TOKENS = 100000000;

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
			node.next_idx = longueur;
			node.token_id = current;
			if (previous != -1)
			{
				node.prev_idx = longueur - 1;

				uint64_t id = pack_pair(previous,current);
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
		if (tas_max.empty()) break;

		HeapNode node;
		uint64_t id;
		do {
		node = tas_max.top();
		tas_max.pop();
		id = node.id;
		} while (node.freq!=freq[id]);

		if (node.freq < 2) break;
	
		merge(id,taille,articles,position,freq, tas_max);
		inverse_vocab[taille] = id;
		taille++;
	}

	ofstream fichier("inverse_vocab.txt");

	if(fichier)  
        {
			for (int i = 256 ; i<taille ; i++)
			{
				uint64_t id = inverse_vocab[i];
				int byte1 = id >> 32;
				int byte2 = id & 0xFFFFFFFF ;
				fichier << i << " " << byte1 << " " << byte2 << "\n";
			}
            
                fichier.close();  
        }
        else 
                cerr << "Erreur à l'ouverture de la sortie !" << endl;
 
    return 0;
}