import { NextRequest, NextResponse } from 'next/server'
import { PineconeClient } from '@pinecone-database/pinecone'// Pinecone
// import { Pinecone } from '@pinecone-database/pinecone'// Pinecone
import { queryPineconeVectorStoreAndQueryLLM } from '@/utils/utils'
import { indexName } from '@/config' // 自己配置的【数据库索引名】


// 🌟 读取数据库的 post 请求, 🌟 在 Next.js 中, API 路由是通过文件系统的结构来定义的 => 如 app/api/read/route.ts 将自动映射到路由 /api/read/route
export async function POST(req: NextRequest) {
	const body = await req.json();
	const client = new PineconeClient()
	await client.init({
		apiKey: process.env.PINECONE_API_KEY || '',
		environment: process.env.PINECONE_ENVIRONMENT || '',
	})
	const text = await queryPineconeVectorStoreAndQueryLLM(client, indexName, body)
	return NextResponse.json({
		data: text
	})
}