import { NextResponse } from 'next/server' // 只需要从数据库中响应内容
import { PineconeClient } from '@pinecone-database/pinecone'// Pinecone
import { TextLoader } from 'langchain/document_loaders/fs/text' // Text 加载器
import { PDFLoader } from 'langchain/document_loaders/fs/pdf' // PDF 加载器
import { DirectoryLoader } from 'langchain/document_loaders/fs/directory' // 文件加载器
import { createPineconeIndex, updatePinecone } from '@/utils/utils'
import { indexName } from '@/config' // 自己配置的【数据库索引名】


// 🌟 修改数据库的 post 请求 => 加载文件 、 把文件 向量化 
export async function POST() {
	// 🔥 实例化文件加载器, 用来加载文本、Markdown、PDF 文件
	const loader = new DirectoryLoader('./documents', { // 🌟 从指定目录（在这里是 './documents'）加载文件 => 配置了三种文件类型的加载器: 文本文件（.txt）、Markdown文件（.md），以及PDF文件（.pdf）
		".txt": (path) => new TextLoader(path),
		".md": (path) => new TextLoader(path),
		".pdf": (path) => new PDFLoader(path),
	})

	const docs = await loader.load()
	const vectorDimension = 1536 // 向量维度

	const client = new PineconeClient()
	await client.init({
		apiKey: process.env.PINECONE_API_KEY || '',
		environment: process.env.PINECONE_ENVIRONMENT || '',
	})

	try {
		// 新增 Pinecone 数据库的一条【索引】
		await createPineconeIndex(client, indexName, vectorDimension) 

		// 修改 Pinecone 数据库
		await updatePinecone(client, indexName, docs) // 把 docs 文档上传上去

	} catch(err) {
		console.log("❌ 报错:", err);
	}

	return NextResponse.json({
		data: "✅ 成功创建将文件创建到 Pinecone 数据库索引内~"
	})
}