import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'
import { OpenAI } from 'langchain/llms/openai'
import { loadQAStuffChain } from 'langchain/chains'
import { timeout } from '../config'

// 创建
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