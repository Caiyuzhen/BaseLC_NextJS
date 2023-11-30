import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'
import { OpenAI } from 'langchain/llms/openai'
import { loadQAStuffChain } from 'langchain/chains'
import { timeout } from '../config'

// 🌟 上传数据到 Pinecone 的方法 ————————————————————————————————————————————————————————————
export const createPineconeIndex = async (
	client,// 数据库客户端
	indexName, // 索引名称
	vectorDimension // 向量数据库
) => {
	// 1. 初始化索引
	console.log(`Checking "${indexName}..."`)

	// 2. 获取现有的索引
	const existingIndexes = await client.listIndexes()

	// 3. 判断索引是否存在
	if(!existingIndexes.includes(indexName)) { // 如果 list 内不存在该索引
		// 4. 不存在，则新建索引
		console.log(`Creating "${indexName}..."`)
		await client.createIndex({
			createRequest: {
				name: indexName,
				dimension: vectorDimension,
				metric: 'cosine', // 余弦相似度
			}
		})
		await new Promise((resolve) => setTimeout(resolve, timeout)) // 用 await nwa Promise 可以更好的处理异步请求 => timeout 为我们自己在 config.js 内定义的 60s
	} else {
		// 5. 存在, 则不做处理
		console.log(`"${indexName}" already exists.`)
	}
}



// 🌟 更新 Pinecone 数据库的方法 ————————————————————————————————————————————————————————————
export const updatePinecone = async (client, indexName, docs) => { //🌟 docs 为要保存在 Pinecone 数据库内的文档
	// 1. 解构赋值, 取回索引
	const index = client.Index(indexName)

	// 2. 映射文档数据
	for(const doc of docs) {
		// 3. 格式化文档内容, 然后才能上传到 Pinecone 数据库内
		const txtPath = doc.metadata.source
		const text = doc.pageContent
		// 4. 创建文本分割器实例
		const textSplitter = new RecursiveCharacterTextSplitter({
			chunkSize: 1000,
		})
		// 5. 使用文本分割器来分割文本, 将句子分成有意义的短语块，如名词短语（NP）或动词短语（VP）
		const chunks = await textSplitter.createDocuments([text])
		console.log(`文本已经分割为: ${chunks.length} chunks, 下一步准备调用 OpenAI 的 embedding`)

		// 6. embedding  => 将分割后的文本、单词、短语表示为连续向量的过程。这些向量捕捉了单词或短语在语义空间中的位置，使得具有相似含义的单词或短语在向量空间中更接近
		const embeddingsArrays = await new OpenAIEmbeddings().embedDocuments(
			chunks.map((chunk) => chunk.pageContent.replace(/\n/g, " ")) // 🌟 把 map 出来的 chunk（分割后的短语）逐个进行向量化
		)

		// 7. 创建一个向量批次: 为了上传数据到 Pinecone, 准备了一个向量批次（batch）。
		const batchSize = 100 // 是用于控制将嵌入向量批量处理的数量, 它定义了每个批次中包含的文本块数量, 处理大量文本数据时, 一次性处理所有文本块可能会导致内存消耗过大或运行时间过长, 使用批处理的方法可以更高效地处理文本块, 因为它允许代码一次性处理一小批文本块, 然后再处理下一批。这可以减轻计算机资源的压力，提高代码的性能
		let batch: any = [] // 用来存储👇👇遍历出来后的 vector 向量
		for(let idx = 0; idx < chunks.length; idx++) {
			const chunk = chunks[idx]
			const vector = {
				id: `${txtPath}_${idx}`,
				values: embeddingsArrays[idx], // 🌟 上方处理后的向量文本
				metadata: {
					...chunk.metadata,
					loc: JSON.stringify(chunk.metadata.loc),
					pageContent: chunk.pageContent,
					txtPath: txtPath,
				}
			}
			batch = [...batch, vector] // 🌟 把新旧数据放在一起

			// 8. 判断是否达到了 batchSize 的限定长度, 或者已经储存完了
			if(batch.length === batchSize || idx === chunks.length - 1) { // idx === chunks.length - 1 是用来判断是否已经遍历到了文本块数组 chunks 的最后一个元素
				await index.upsert({
					upsertRequest: {
						vectors: batch,
					}
				})
				// 如果储存完后, 就清空 batch
				batch = [] 
			}
		}
	}
}
