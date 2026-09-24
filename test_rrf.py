from app.retrieval import reciprocal_rank_fusion

def main():
    vector_results = ["chunks_A","chunks_B","chunks_C"]
    keyword_results = ["chunks_C","chunks_D","chunks_A"]


    fused = reciprocal_rank_fusion(vector_results,keyword_results)
    print("Fused ranking",fused)

if __name__ =="__main__":
    main()    

